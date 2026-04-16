"""Convenience entrypoint for running the classic methods pipeline once."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline import Pipeline
from validation import ValidationPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline runner."""

    parser = argparse.ArgumentParser(description="Run the classic methods pipeline.")
    parser.add_argument(
        "--config",
        default="config/base.yaml",
        help="Path to the root pipeline config relative to the classic_methods directory.",
    )
    return parser.parse_args()


def main(config_path: Path | None = None) -> None:
    """Run the configured pipeline and print the output location."""

    args = parse_args()
    if config_path is None:
        config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()

    raw_config = Pipeline._load_config(config_path)
    if bool(raw_config.get("validation_bool", False)):
        validation_pipeline = ValidationPipeline(config_path)
        summary = validation_pipeline.run()
        print(f"Validation summary: {validation_pipeline.config.summary_path}")
        print(f"Repetitions run: {summary['repetitions_run']}")
        return

    pipeline = Pipeline(config_path)
    artifacts = pipeline.run()
    print(f"Output directory: {pipeline.output_dir}")
    print(f"Artifacts: {', '.join(sorted(artifacts))}")


if __name__ == "__main__":
    #config_path = ROOT / "config" / "base.yaml"
    config_path = None
    main(config_path)
