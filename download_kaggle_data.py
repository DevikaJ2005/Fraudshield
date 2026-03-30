#!/usr/bin/env python3
"""Download the Kaggle source CSV and regenerate the compact FraudShield bundle."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from data_loader import KaggleDataLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_kaggle_setup() -> bool:
    """Validate that the local Kaggle token exists."""

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        logger.info("Found %s", kaggle_json)
        return True

    logger.error("kaggle.json was not found at %s", kaggle_json)
    logger.error("Create a Kaggle API token and place it there before running this script.")
    return False


def main() -> int:
    """Download or refresh the source dataset, then rebuild the task bundle."""

    if not check_kaggle_setup():
        return 1

    loader = KaggleDataLoader(data_path="data", seed=42)
    if not loader.download_data():
        return 1

    if loader.bundle_file.exists():
        loader.bundle_file.unlink()

    if not loader.load_data():
        return 1

    logger.info("Task bundle is ready at %s", loader.bundle_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
