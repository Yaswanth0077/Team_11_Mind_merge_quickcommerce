"""Command-line entry point for the Phase 1 audit and data-preparation pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from config import (
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    REPORT_DIR,
    SLA_THRESHOLD_MINUTES,
)
from src.data.cleaning import clean_orders
from src.data.feature_engineering import engineer_features
from src.data.loader import load_orders
from src.data.validation import (
    add_quality_flags,
    build_data_quality_report,
    write_report,
)


@dataclass(frozen=True)
class PipelineResult:
    """Paths and row counts produced by a successful data-pipeline run."""

    processed_path: Path
    json_report_path: Path
    markdown_report_path: Path
    input_records: int
    retained_records: int
    excluded_records: int


def run_data_pipeline(
    input_path: Path = RAW_DATA_PATH,
    processed_path: Path = PROCESSED_DATA_PATH,
    report_dir: Path = REPORT_DIR,
    sla_threshold_minutes: int = SLA_THRESHOLD_MINUTES,
) -> PipelineResult:
    """Load, audit, conservatively clean, engineer, and export the dataset."""
    if sla_threshold_minutes <= 0:
        raise ValueError("SLA threshold must be a positive number of minutes.")
    raw = load_orders(input_path)
    cleaned, cleaning_log = clean_orders(raw)
    enriched = engineer_features(cleaned, sla_threshold_minutes)
    enriched, flag_log = add_quality_flags(enriched)
    report = build_data_quality_report(
        raw,
        cleaned,
        enriched,
        cleaning_log,
        flag_log,
        input_path,
        sla_threshold_minutes,
    )

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(processed_path, index=False, na_rep="")
    json_path, markdown_path = write_report(report, report_dir)
    return PipelineResult(
        processed_path=processed_path,
        json_report_path=json_path,
        markdown_report_path=markdown_path,
        input_records=len(raw),
        retained_records=len(enriched),
        excluded_records=len(raw) - len(enriched),
    )


def main() -> None:
    """Run the configured pipeline or accept a custom source CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DATA_PATH,
        help="Path to the original CSV (default: data/raw/train.csv).",
    )
    args = parser.parse_args()
    result = run_data_pipeline(input_path=args.input)
    print(f"Processed dataset: {result.processed_path}")
    print(f"Markdown report: {result.markdown_report_path}")
    print(f"Structured report: {result.json_report_path}")
    print(
        f"Records: {result.retained_records:,} retained; "
        f"{result.excluded_records:,} excluded."
    )


if __name__ == "__main__":
    main()
