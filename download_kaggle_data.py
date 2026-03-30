#!/usr/bin/env python3
"""Refresh the local FraudShield snapshot from the public source dataset."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from data_loader import FraudDataLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_kaggle_setup() -> bool:
    """Validate that the local Kaggle token exists when a download is needed."""

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        logger.info("Found %s", kaggle_json)
        return True

    logger.error("kaggle.json was not found at %s", kaggle_json)
    logger.error("Create a Kaggle API token and place it there before downloading source data.")
    return False


def main() -> int:
    """Rebuild the FraudShield snapshot from the local CSV or download it first if missing."""

    loader = FraudDataLoader(data_path="data", seed=42)

    if not loader.csv_file.exists():
        if not check_kaggle_setup():
            return 1
        if not loader.download_source_data():
            return 1
    else:
        logger.info("Found existing public source CSV at %s", loader.csv_file)

    if loader.bundle_file.exists():
        loader.bundle_file.unlink()

    if not loader.load_bundle():
        return 1

    logger.info("Snapshot summary: %s", loader.get_bundle_summary())
    logger.info("FraudShield snapshot is ready at %s", loader.bundle_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
