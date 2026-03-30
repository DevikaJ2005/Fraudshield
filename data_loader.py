"""
FraudShield Data Loader
Loads and processes Kaggle Credit Card Fraud Detection dataset
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class KaggleDataLoader:
    """Load and process Kaggle credit card fraud dataset"""

    def __init__(self, data_path: str = "data"):
        """Initialize data loader"""
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)
        self.csv_file = self.data_path / "creditcard.csv"
        self.df = None

    def download_data(self) -> bool:
        """Download Kaggle dataset"""
        try:
            import kaggle
            logger.info("Downloading Kaggle Credit Card Fraud dataset...")
            
            kaggle.api.dataset_download_files(
                "mlg-ulb/creditcardfraud",
                path=str(self.data_path),
                unzip=True
            )
            
            logger.info(f"✓ Data downloaded to {self.data_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download: {e}")
            logger.error("Make sure kaggle.json is in ~/.kaggle/")
            return False

    def load_data(self) -> bool:
        """Load CSV data"""
        if not self.csv_file.exists():
            logger.error(f"Data file not found: {self.csv_file}")
            logger.error("Run download_data() first")
            return False

        try:
            logger.info(f"Loading data from {self.csv_file}...")
            self.df = pd.read_csv(self.csv_file)
            logger.info(f"✓ Loaded {len(self.df)} transactions")
            logger.info(f"  Fraud count: {(self.df['Class'] == 1).sum()}")
            logger.info(f"  Legitimate count: {(self.df['Class'] == 0).sum()}")
            return True
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False

    def get_fraud_transactions(self, count: int = 30) -> List[Dict]:
        """Get fraud transactions"""
        if self.df is None:
            return []
        
        fraud_df = self.df[self.df['Class'] == 1].head(count)
        return self._convert_to_transactions(fraud_df, label="fraud")

    def get_legitimate_transactions(self, count: int = 30) -> List[Dict]:
        """Get legitimate transactions"""
        if self.df is None:
            return []
        
        legit_df = self.df[self.df['Class'] == 0].head(count)
        return self._convert_to_transactions(legit_df, label="legitimate")

    def _convert_to_transactions(self, df: pd.DataFrame, label: str) -> List[Dict]:
        """Convert DataFrame rows to transaction dicts"""
        transactions = []
        
        for idx, row in df.iterrows():
            # Use PCA features V1-V28 (from Kaggle dataset)
            amount = float(row['Amount'])
            
            txn = {
                "transaction_id": f"{label}_{idx}",
                "amount": amount,
                "seller_id": f"card_{hash(str(row['Time'])) % 10000}",
                "buyer_id": f"buyer_{idx}",
                "item_category": self._infer_category(amount),
                "item_price": amount * 0.9,  # Assume 90% of amount is item
                "shipping_address": self._infer_country(row),
                "seller_account_age_days": np.random.randint(1, 3650),
                "buyer_account_age_days": np.random.randint(1, 3650),
                "payment_method": "credit_card",
                "device_country": "US",
                "timestamp": f"2024-03-{(idx % 28) + 1:02d}T10:30:00",
                "is_repeat_buyer": np.random.choice([True, False], p=[0.6, 0.4]),
                "seller_avg_rating": np.random.uniform(3.5, 5.0),
                "num_seller_reviews": np.random.randint(10, 5000),
                "previous_fraud_flags": 1 if label == "fraud" else 0,
                "pca_features": row[[f'V{i}' for i in range(1, 29)]].values.tolist(),
            }
            transactions.append(txn)
        
        return transactions

    def _infer_category(self, amount: float) -> str:
        """Infer item category from amount"""
        if amount < 50:
            return "groceries"
        elif amount < 200:
            return "electronics"
        elif amount < 500:
            return "clothing"
        elif amount < 1000:
            return "jewelry"
        else:
            return "collectibles"

    def _infer_country(self, row: pd.Series) -> str:
        """Infer shipping country from features"""
        # Use PCA features to probabilistically assign country
        v1_val = row.get('V1', 0)
        
        if v1_val < -1.5:
            return np.random.choice(["NG", "RU", "CN"], p=[0.4, 0.3, 0.3])
        elif v1_val > 2.0:
            return np.random.choice(["US", "GB", "CA"], p=[0.5, 0.3, 0.2])
        else:
            return np.random.choice(["US", "GB", "CA", "AU", "IN"], p=[0.4, 0.2, 0.2, 0.1, 0.1])

    def get_split_by_difficulty(self) -> Tuple[List[Dict], List[Dict], List[Dict], 
                                                List[str], List[str], List[str]]:
        """
        Get data split into 3 difficulty levels
        Returns: easy_txns, medium_txns, hard_txns, easy_labels, medium_labels, hard_labels
        """
        if self.df is None:
            logger.error("Data not loaded. Call load_data() first")
            return [], [], [], [], [], []

        # Split by difficulty
        easy_fraud = self.get_fraud_transactions(30)
        easy_legit = self.get_legitimate_transactions(30)
        easy = easy_fraud + easy_legit
        easy_labels = ["fraud"] * 30 + ["legitimate"] * 30

        medium_fraud = self.get_fraud_transactions(50)
        medium_legit = self.get_legitimate_transactions(50)
        medium = medium_fraud + medium_legit
        medium_labels = ["fraud"] * 50 + ["legitimate"] * 50

        hard_fraud = self.get_fraud_transactions(100)
        hard_legit = self.get_legitimate_transactions(100)
        hard = hard_fraud + hard_legit
        hard_labels = ["fraud"] * 100 + ["legitimate"] * 100

        # Shuffle
        import random
        for data, labels in [(easy, easy_labels), (medium, medium_labels), (hard, hard_labels)]:
            combined = list(zip(data, labels))
            random.shuffle(combined)
            data[:], labels[:] = zip(*combined)

        logger.info(f"✓ Data split:")
        logger.info(f"  Easy: {len(easy)} (fraud: {easy_labels.count('fraud')})")
        logger.info(f"  Medium: {len(medium)} (fraud: {medium_labels.count('fraud')})")
        logger.info(f"  Hard: {len(hard)} (fraud: {hard_labels.count('fraud')})")

        return easy, medium, hard, easy_labels, medium_labels, hard_labels
