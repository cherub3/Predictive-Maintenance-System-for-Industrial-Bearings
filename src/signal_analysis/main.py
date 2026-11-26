"""Phase 3: Signal Understanding & Exploratory Signal Analysis — Pipeline Entry Point.

Orchestrates the complete Phase 3 pipeline:
  1. Load Phase 1 metadata
  2. Compute Tier 1 time-domain health indicators (all 2,156 files x 8 channels)
  3. Select Tier 2 spectral checkpoints
  4. Run Tier 2 spectral analysis
  5. Generate visualizations
  6. Resolve feature gate decisions
  7. Write phase3_gating_decisions.yaml
  8. Generate engineering findings and analysis report

Exits with EXIT_SUCCESS on completion or EXIT_ERROR_SIGNAL_ANALYSIS on failure.
"""

import sys
import time
from typing import List, Optional

import pandas as pd

from src.config import load_config
from src.constants import (
    DEFAULT_CONFIG_PATH,
    EXIT_SUCCESS,
    EXIT_ERROR_CONFIG,
    EXIT_ERROR_SIGNAL_ANALYSIS,
    PROJECT_ROOT,
)
from src.logger import setup_logger
from src.utils import ensure_directory_exists

from src.signal_analysis.bearing_geometry import (
    DefectFrequencies,
    load_bearing_geometry,
    compute_defect_frequencies,
)
from src.signal_analysis.health_indicators import (
    compute_health_indicators,
    validate_health_indicators,
)
from src.signal_analysis.checkpoint_selector import select_checkpoints
from src.signal_analysis.gating_contract import (
    build_gating_contract,
    write_gating_contract,
)
from src.signal_analysis.engineering_findings import generate_engineering_findings
from src.signal_analysis.report_generator import generate_signal_analysis_report
from src.signal_analysis.visualization.health_trend_plots import (
    generate_all_health_trend_plots,
)
from src.signal_analysis.visualization.spectral_plots import (
    plot_spectral_evolution,
    plot_band_energy_evolution,
)


def run_signal_analysis_pipeline() -> int:
    """Execute the complete Phase 3 signal analysis pipeline.

    Returns:
        int: EXIT_SUCCESS on success, EXIT_ERROR_SIGNAL_ANALYSIS on failure.
    """
    pipeline_start = time.time()

    # ── Configuration ────────────────────────────────────────────────────────
    try:
        config = load_config(DEFAULT_CONFIG_PATH)
    except Exception as exc:
        print(f"Configuration load failed: {exc}")
        return EXIT_ERROR_CONFIG

    logger = setup_logger(name="signal_analysis", config=config)
    logger.info("=" * 80)
    logger.info("PHASE 3 — SIGNAL UNDERSTANDING & EXPLORATORY SIGNAL ANALYSIS")
    logger.info("=" * 80)

    try:
        paths_cfg = config["paths"]
        sa_cfg = config["signal_analysis"]
        sa_output = sa_cfg["output_paths"]

        raw_data_dir = PROJECT_ROOT / paths_cfg["raw_data_dir"]
        processed_dir = PROJECT_ROOT / paths_cfg["processed_data_dir"]
        report_dir = PROJECT_ROOT / paths_cfg["report_dir"]
        figures_dir = PROJECT_ROOT / sa_output["figures_dir"]

        ensure_directory_exists(processed_dir)
        ensure_directory_exists(report_dir)
        ensure_directory_exists(figures_dir)

        # ── Step 1: Load Phase 1 metadata ────────────────────────────────────
        logger.info("Step 1: Loading Phase 1 metadata")
        metadata_path = processed_dir / "metadata.parquet"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Phase 1 metadata not found: {metadata_path}")

        metadata_df = pd.read_parquet(metadata_path)
        logger.info(f"Loaded {len(metadata_df)} metadata records")

        # ── Step 2: Load bearing geometry ────────────────────────────────────
        logger.info("Step 2: Loading bearing geometry")
        bearing_geometry_verified = False
        defect_freqs: Optional[DefectFrequencies] = None

        try:
            geometry = load_bearing_geometry(config)
            defect_freqs = compute_defect_frequencies(geometry)
            bearing_geometry_verified = True
            logger.info(
                f"Bearing geometry loaded: BPFO={defect_freqs.bpfo_hz:.1f} Hz, "
                f"BPFI={defect_freqs.bpfi_hz:.1f} Hz, BSF={defect_freqs.bsf_hz:.1f} Hz"
            )
        except Exception as exc:
            logger.warning(f"Bearing geometry load failed: {exc}. Envelope features will be excluded.")

        # ── Step 3: Tier 1 — Health indicators ──────────────────────────────
        logger.info("Step 3: Computing Tier 1 time-domain health indicators")
        health_df = compute_health_indicators(
            metadata_df, config, logger
        )

        # Each dataset has its own channel count (8 for set1, 4 for set2/set3), so
        # the expected row count is the per-file channel count summed, not a single
        # global constant times file count.
        success_meta = metadata_df[metadata_df["parser_status"] == "Success"]
        expected_rows = int(success_meta["expected_channels_per_file"].sum())
        if not validate_health_indicators(health_df, expected_rows, logger):
            raise ValueError("Health indicator validation failed")

        # Save health indicators
        indicators_path = PROJECT_ROOT / sa_output["health_indicators"]
        indicators_path.parent.mkdir(parents=True, exist_ok=True)
        health_df.to_parquet(indicators_path, index=False)
        logger.info(f"Health indicators saved: {indicators_path} ({len(health_df):,} rows)")

        # ── Step 4: Tier 2 — Checkpoint selection ────────────────────────────
        logger.info("Step 4: Selecting Tier 2 spectral checkpoints")
        checkpoint_df = select_checkpoints(health_df, config, logger)

        checkpoint_path = PROJECT_ROOT / sa_output["checkpoint_selection"]
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_df.to_csv(checkpoint_path, index=False)
        logger.info(f"Checkpoints saved: {checkpoint_path} ({len(checkpoint_df)} files)")

        # Enrich an in-memory copy with each file's already-resolved absolute_path
        # (from Phase 1 metadata) for the plotting step below — datasets now live
        # in per-dataset raw subfolders, so a single global raw_dir join no longer
        # resolves correctly and the persisted CSV schema stays unchanged.
        checkpoint_df_with_paths = checkpoint_df.merge(
            metadata_df[["dataset_id", "filename", "absolute_path"]],
            on=["dataset_id", "filename"],
            how="left",
        )

        # ── Step 5: Visualizations ───────────────────────────────────────────
        logger.info("Step 5: Generating visualizations")
        figures_generated: List[str] = []

        try:
            generate_all_health_trend_plots(health_df, config, figures_dir, logger)
            # Collect generated figure names for the report
            for fig_file in sorted(figures_dir.rglob("*.png")):
                figures_generated.append(str(fig_file.relative_to(PROJECT_ROOT)))
        except Exception as exc:
            logger.warning(f"Health trend plots failed (non-fatal): {exc}")

        try:
            plot_spectral_evolution(
                checkpoint_df_with_paths, config, defect_freqs, figures_dir, logger
            )
            plot_band_energy_evolution(
                health_df, checkpoint_df_with_paths, config, figures_dir, logger
            )
            for fig_file in sorted(figures_dir.rglob("*.png")):
                rel = str(fig_file.relative_to(PROJECT_ROOT))
                if rel not in figures_generated:
                    figures_generated.append(rel)
        except Exception as exc:
            logger.warning(f"Spectral plots failed (non-fatal): {exc}")

        logger.info(f"Generated {len(figures_generated)} figures")

        # ── Step 6: Resolve feature gates ────────────────────────────────────
        logger.info("Step 6: Resolving feature gate decisions")
        gate_contract = build_gating_contract(
            health_df, config, bearing_geometry_verified, logger
        )

        # ── Step 7: Write gating contract (Phase 3 → Phase 4 interface) ─────
        logger.info("Step 7: Writing Phase 3 → Phase 4 gate contract")
        contract_path = PROJECT_ROOT / sa_output["gating_contract"]
        write_gating_contract(gate_contract, contract_path, logger)

        # ── Step 8: Generate human-readable documents ────────────────────────
        logger.info("Step 8: Generating engineering findings and report")

        findings_path = PROJECT_ROOT / sa_output["engineering_findings"]
        generate_engineering_findings(
            health_df,
            checkpoint_df,
            gate_contract,
            bearing_geometry_verified,
            config,
            findings_path,
            logger,
        )

        elapsed = time.time() - pipeline_start
        report_path = PROJECT_ROOT / sa_output["signal_analysis_report"]
        generate_signal_analysis_report(
            health_df,
            checkpoint_df,
            gate_contract,
            bearing_geometry_verified,
            config,
            figures_generated,
            elapsed,
            report_path,
            logger,
        )

        # ── Final summary ─────────────────────────────────────────────────────
        logger.info("=" * 80)
        logger.info("PHASE 3 SIGNAL ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Health indicators:   {indicators_path}")
        logger.info(f"Checkpoints:         {checkpoint_path} ({len(checkpoint_df)} files)")
        logger.info(f"Gate contract:       {contract_path}")
        logger.info(f"Engineering findings:{findings_path}")
        logger.info(f"Analysis report:     {report_path}")
        logger.info(f"Figures generated:   {len(figures_generated)}")
        logger.info(f"Elapsed time:        {elapsed:.1f} seconds")

        gate_summary = {k: v["include"] for k, v in gate_contract["gate_decisions"].items()}
        logger.info(f"Gate decisions:      {gate_summary}")

        return EXIT_SUCCESS

    except FileNotFoundError as exc:
        logger.error(f"Required file not found: {exc}")
        return EXIT_ERROR_SIGNAL_ANALYSIS
    except Exception as exc:
        logger.error(f"Phase 3 pipeline failed: {exc}", exc_info=True)
        return EXIT_ERROR_SIGNAL_ANALYSIS


if __name__ == "__main__":
    exit_code = run_signal_analysis_pipeline()
    sys.exit(exit_code)
