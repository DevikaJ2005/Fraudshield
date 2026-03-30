#!/usr/bin/env python3
"""
Download Kaggle Credit Card Fraud Detection Dataset
Run this once to download data for the project

Usage:
    python download_kaggle_data.py
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_kaggle_setup():
    """Check if Kaggle is properly configured"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        logger.error("❌ kaggle.json not found!")
        logger.error(f"   Expected location: {kaggle_json}")
        logger.error("\n📋 Setup Instructions:")
        logger.error("   1. Go to: https://www.kaggle.com/settings/account")
        logger.error("   2. Scroll to 'API' section")
        logger.error("   3. Click 'Create New API Token'")
        logger.error("   4. This downloads kaggle.json")
        logger.error(f"   5. Move it to: {kaggle_dir}/")
        return False
    
    logger.info("✓ kaggle.json found")
    return True


def download_data():
    """Download Kaggle dataset"""
    try:
        import kaggle
        from data_loader import KaggleDataLoader
        
        logger.info("\n" + "="*70)
        logger.info("📥 Downloading Kaggle Credit Card Fraud Dataset")
        logger.info("="*70)
        
        loader = KaggleDataLoader(data_path="data")
        
        if loader.download_data():
            logger.info("\n" + "="*70)
            logger.info("✓ Download successful!")
            logger.info("="*70)
            
            if loader.load_data():
                logger.info("\n✓ Data loaded successfully!")
                logger.info(f"  Total transactions: {len(loader.df)}")
                logger.info(f"  Frauds: {(loader.df['Class'] == 1).sum()}")
                logger.info(f"  Legitimate: {(loader.df['Class'] == 0).sum()}")
                logger.info("\n✓ Ready to run inference!")
                return True
        
        return False
        
    except ImportError:
        logger.error("❌ kaggle package not installed!")
        logger.error("   Run: pip install kaggle")
        return False
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False


if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("🔍 Checking Kaggle Setup")
    logger.info("="*70)
    
    if not check_kaggle_setup():
        sys.exit(1)
    
    if not download_data():
        sys.exit(1)
    
    logger.info("\n✓ All set! Run: python inference.py\n")
