"""Master execution pipeline for REFIT NILM project.

Stages:
  - download: Download raw and clean 7z archives if not present
  - prepare:  Prepare 1-minute Parquet files for all cleaned houses
  - section1: Raw House 1 quality audit, cleaning, figures, and DATA.md draft
  - quality:  Quality masks, stale runs, cycle tuning, and eligibility freeze
  - eda:      Washing machine exploratory data analysis and figures
  - model:    LightGBM NILM disaggregation model, baselines, and evaluation
  - docs:     Build markdown tables and update documentation
  - all:      Run all stages end-to-end
"""

from pathlib import Path
import sys
import argparse
import time

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config as C


def stage_download():
    """Stage: Download archives."""
    from scripts import download_data
    download_data.main()


def stage_prepare():
    """Stage: Prepare cleaned 1-min parquets."""
    from scripts import prepare_data
    prepare_data.main()


def stage_section1():
    """Stage: Raw House 1 inspection, cleaning, figures, and documentation."""
    from src.cleaning import load_house1_raw, inspect_house1_raw, clean_house1_raw, plot_house1_cleaning, write_data_md_section1

    print("=" * 60)
    print("STAGE: Section 1 - Raw House 1 Inspection & Cleaning")
    print("=" * 60)
    t0 = time.time()

    # 1. Load raw
    print("Loading raw House 1 parts...")
    df1, df2, concat_df = load_house1_raw()

    # 2. Inspect
    print("Inspecting raw House 1 data...")
    summary_metrics = inspect_house1_raw(df1, df2, concat_df)

    # 3. Clean
    print("Executing cleaning pipeline...")
    cleaned_df, outages_df = clean_house1_raw(concat_df)

    # 4. Plots
    print("Generating diagnostic plots...")
    plot_house1_cleaning(concat_df, cleaned_df, outages_df)

    # 5. Documentation
    print("Writing DATA.md Section 1 draft...")
    write_data_md_section1(summary_metrics, outages_df)

    print(f"\nStage Section 1 completed in {time.time() - t0:.1f}s")
    print("=" * 60)


def stage_quality():
    """Stage: Quality masks, stale check, cycle tuning, availability and eligibility."""
    from src.quality import stage_quality as run_quality
    run_quality()


def stage_eda():
    """Stage: Washing machine exploratory data analysis, summary, and figures."""
    from src.eda import stage_eda as run_eda
    run_eda()


def main():
    parser = argparse.ArgumentParser(description="Run REFIT NILM pipeline stages.")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["download", "prepare", "section1", "quality", "eda", "model", "docs", "all"],
        help="Stage to execute",
    )
    args = parser.parse_args()

    if args.stage == "download":
        stage_download()
    elif args.stage == "prepare":
        stage_prepare()
    elif args.stage == "section1":
        stage_section1()
    elif args.stage == "quality":
        stage_quality()
    elif args.stage == "eda":
        stage_eda()
    elif args.stage == "all":
        stage_download()
        stage_prepare()
        stage_section1()
        stage_quality()
        stage_eda()
        # Future stages will be called here as they are implemented
    else:
        print(f"Stage '{args.stage}' not yet implemented or unknown.")


if __name__ == "__main__":
    main()
