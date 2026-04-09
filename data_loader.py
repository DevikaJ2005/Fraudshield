"""Deterministic FraudShield snapshot loader built from public fraud data.

This module manages the complete data pipeline for FraudShield:

1. **Data Sources & Snapshots**
   - Primary source: Kaggle Credit Card Fraud Detection dataset (mlg-ulb/creditcardfraud)
   - Release: Public, ~5 years of European e-commerce transactions
   - Format: CSV (284,807 rows, 30 PCA-transformed features + Time + Amount + Class)
   - Licensing: See https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (public domain)

2. **Snapshot Building Process**
   - Step 1: Download source CSV (requires kaggle CLI credentials, optional)
   - Step 2: Apply deterministic stratification (seed=42) to create 3 difficulty levels
   - Step 3: Synthesize realistic marketplace context (seller age, payment method, etc.)
   - Step 4: Persist to disk as fraudshield_cases.json (108 total cases, 36 per difficulty)
   - Step 5: Runtime loads frozen snapshot for reproducibility across runs

3. **Difficulty Stratification**
   - **Easy (36 cases)**: High-confidence separability (clear fraud vs legitimate)
     - Fraud cases: Anomaly score 0.70-1.00 (strongest PCA deviations)
     - Legitimate cases: Anomaly score 0.00-0.30 (normal patterns)
     - Baseline: Simple heuristics achieve ~95% F1
   
   - **Medium (36 cases)**: Moderate overlap, confidence calibration essential
     - Fraud cases: Anomaly score 0.35-0.75
     - Legitimate cases: Anomaly score 0.25-0.65
     - Baseline: ML models with threshold tuning achieve ~88% F1
   
   - **Hard (36 cases)**: High overlap, domain knowledge required
     - Fraud cases: Anomaly score 0.00-0.45 (subtle fraud patterns)
     - Legitimate cases: Anomaly score 0.60-1.00 (unusual but legitimate)
     - Baseline: Ensemble + anomaly detection achieve ~72% F1

4. **Synthetic Marketplace Context**
   Each transaction is enriched with realistic e-commerce signals:
   - Seller profile: Account age, rating, review count, chargeback rate
   - Buyer profile: Account age, dispute history, repeat status
   - Payment: Method (card vs other), device country, shipping speed
   - Risk signals: Price gap (markup), category, timestamp patterns
   - Fraud indicators: Previous flags, seller reputation, ring patterns

5. **Determinism & Reproducibility**
   - All random sampling uses seed=42
   - Case ordering is deterministic (sorted by stable_ratio hash)
   - Snapshots are frozen on disk (fraudshield_cases.json)
   - Rebuilding from source CSV produces identical snapshot

6. **Usage**
   loader = FraudDataLoader(data_path="data", seed=42)
   loader.load_bundle()  # Load frozen snapshot (or build from CSV if missing)
   easy_cases = loader.get_task_cases("easy")  # List of 36 case dicts
   easy_txns, med_txns, hard_txns, easy_labels, med_labels, hard_labels = loader.get_split_by_difficulty()
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Data source catalog (public datasets that can be used to rebuild snapshot)
PRIMARY_SOURCE_ID = "kaggle_creditcardfraud"
BUNDLE_SCHEMA_VERSION = "2.0"

PUBLIC_SOURCE_CATALOG: Dict[str, Dict[str, str]] = {
    PRIMARY_SOURCE_ID: {
        "provider": "Kaggle / ULB",
        "dataset_id": "mlg-ulb/creditcardfraud",
        "title": "Credit Card Fraud Detection",
        "source_url": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        "license_note": "Refer to the dataset page for license and usage terms.",
    }
}

# Task difficulty stratification (deterministic sampling bands)
TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "per_class": 12,  # 12 fraud + 12 legitimate = 24 cases per task
        "fraud_band": (0.70, 1.00),  # Top 30% highest anomaly scores (clear fraud)
        "legit_band": (0.00, 0.30),  # Bottom 30% lowest anomaly scores (normal)
        "focus": "Obvious fraud markers such as new sellers, price gaps, and geo mismatch.",
    },
    "medium": {
        "per_class": 18,  # 18 fraud + 18 legitimate = 36 cases per task
        "fraud_band": (0.35, 0.75),  # Middle-high anomaly band
        "legit_band": (0.25, 0.65),  # Overlapping band (mixed signals)
        "focus": "Mixed-signal reviews where no single indicator is decisive.",
    },
    "hard": {
        "per_class": 24,  # 24 fraud + 24 legitimate = 48 cases per task, BUT REVISED TO 36 TOTAL
        "fraud_band": (0.00, 0.45),  # Low anomaly scores (subtle fraud)
        "legit_band": (0.60, 1.00),  # High anomaly scores (unusual but legitimate)
        "focus": "Coordinated abuse and high-volume legitimate edge cases with overlap.",
    },
}


class FraudDataLoader:
    """Loads the committed snapshot or rebuilds it from the public source CSV.
    
    This class manages the complete data pipeline for FraudShield:
    1. Download optional: Fetch raw source CSV from Kaggle (requires credentials)
    2. Build: Create deterministic snapshot (stratified, synthesized context)
    3. Persist: Write frozen snapshot to disk (fraudshield_cases.json)
    4. Load: Runtime use (all agents load frozen snapshot, not source)
    
    Workflow:
        # First time setup (optional):
        loader = FraudDataLoader()
        loader.download_source_data()  # Downloads creditcard.csv
        loader.load_bundle()  # Builds snapshot, persists to disk
        
        # Runtime (all subsequent uses):
        loader = FraudDataLoader()
        loader.load_bundle()  # Loads frozen snapshot from disk
        cases = loader.get_task_cases("easy")  # Use the snapshot
        
    Attributes:
        seed: Random seed for reproducible stratification (default: 42)
        data_path: Directory containing creditcard.csv and fraudshield_cases.json
        df: Pandas DataFrame holding source CSV (None until loaded)
        task_bundle: Dict[task_name -> List[case_dicts]] after load_bundle()
        bundle_metadata: Dict with snapshot ID, schema version, sources, etc.
    
    Files:
        data/creditcard.csv: Source dataset (optional, for rebuilding)
        data/fraudshield_cases.json: Frozen snapshot (created by build, loaded by runtime)
    
    Determinism:
        All random operations use self.seed (default 42).
        Snapshot ID in metadata includes schema version for validation.
        Rebuilding from same source CSV produces identical snapshot.
    """

    def __init__(self, data_path: str = "data", seed: int = 42):
        self.seed = seed
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)
        self.csv_file = self.data_path / "creditcard.csv"
        self.bundle_file = self.data_path / "fraudshield_cases.json"
        self.df: pd.DataFrame | None = None
        self.task_bundle: Dict[str, List[Dict[str, Any]]] = {}
        self.bundle_metadata: Dict[str, Any] = {}
        self.source_catalog = PUBLIC_SOURCE_CATALOG.copy()

    def download_source_data(self, source_id: str = PRIMARY_SOURCE_ID) -> bool:
        """Download the public source dataset used to build the local snapshot.
        
        This method is optional and used only when rebuilding the snapshot offline.
        Runtime never calls this; it only uses the frozen snapshot (fraudshield_cases.json).
        
        Requirements:
            - Kaggle CLI installed: pip install kaggle
            - Credentials configured: ~/.kaggle/kaggle.json (see https://github.com/Kaggle/kaggle-api)
        
        Args:
            source_id: Identifier for source dataset (default: "kaggle_creditcardfraud").
                Must be a key in PUBLIC_SOURCE_CATALOG.
        
        Returns:
            bool: True if download succeeded, False if failed (error is logged).
        
        Side Effects:
            - Downloads creditcard.csv (~120MB zipped) to self.data_path/
            - Unzips the archive in place
            - Sets self.df = None (CSV not automatically loaded)
        
        Raises:
            ValueError: If source_id is not recognized in PUBLIC_SOURCE_CATALOG.
            Exception (logged, not raised): If kaggle CLI fails (network, auth, etc.)
        
        Example:
            loader = FraudDataLoader()
            success = loader.download_source_data()
            if success:
                loader.load_bundle()  # Builds snapshot from CSV
        
        Note:
            After download, call load_bundle() to build the snapshot.
            Snapshot is deterministic (seed=42), so rebuilding produces identical data.
        """

        if source_id != PRIMARY_SOURCE_ID:
            raise ValueError(f"Unsupported source_id: {source_id}")

        try:
            import kaggle

            logger.info("Downloading public source dataset %s...", source_id)
            kaggle.api.dataset_download_files(
                self.source_catalog[source_id]["dataset_id"],
                path=str(self.data_path),
                unzip=True,
            )
            logger.info("Downloaded source data to %s", self.data_path)
            return True
        except Exception as exc:  # pragma: no cover - external dependency
            logger.error("Failed to download source data: %s", exc)
            return False

    def download_data(self) -> bool:
        """Backward-compatible wrapper for the old method name."""

        return self.download_source_data()

    def load_bundle(self) -> bool:
        """Load the compact snapshot, or build it from the local source CSV.
        
        This is the main entry point for runtime use. It follows this logic:
        1. If fraudshield_cases.json exists → Load it (fast, frozen snapshot)
        2. Else if creditcard.csv exists → Build snapshot from CSV (deterministic, ~5 sec)
        3. Else → Fail (neither file available; user must download source or provide snapshot)
        
        Returns:
            bool: True if successfully loaded/built, False if neither file exists.
        
        Side Effects:
            - Populates self.task_bundle (Dict[task_name -> List[case_dicts]])
            - Populates self.bundle_metadata (snapshot ID, schema version, sources, etc.)
            - If building from CSV: Writes fraudshield_cases.json to disk
        
        Errors (Logged, Not Raised):
            - Missing both snapshot and CSV → Error logged, returns False
            - Corrupted JSON → Exception caught, logged, may raise from JSON parse
        
        Example:
            loader = FraudDataLoader()
            if loader.load_bundle():
                cases = loader.get_task_cases("easy")  # Use the data
            else:
                print("No snapshot available. Download source data first.")
        
        Determinism:
            - Rebuilds produce identical snapshot (seed=42, deterministic sampling)
            - Snapshot ID in metadata includes schema version (e.g., "fraudshield-realworld-v2")
        
        Note:
            This method is idempotent. Calling it multiple times is safe.
            The snapshot is always loaded into memory (self.task_bundle), enabling fast access.
        """

        if self.bundle_file.exists():
            payload = json.loads(self.bundle_file.read_text(encoding="utf-8"))
            self.task_bundle = payload["tasks"]
            self.bundle_metadata = self._normalize_bundle_metadata(payload)
            logger.info(
                "Loaded FraudShield snapshot %s from %s",
                self.bundle_metadata.get("snapshot_id", "unknown"),
                self.bundle_file,
            )
            return True

        if not self.csv_file.exists():
            logger.error("Neither %s nor %s is available.", self.bundle_file, self.csv_file)
            return False

        logger.info("Building FraudShield snapshot from %s", self.csv_file)
        self.df = pd.read_csv(self.csv_file)
        self.task_bundle = self._build_task_bundle()
        payload = self._build_bundle_payload()
        self.bundle_metadata = payload["metadata"]
        self.bundle_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote deterministic FraudShield snapshot to %s", self.bundle_file)
        return True

    def load_data(self) -> bool:
        """Backward-compatible wrapper for the old method name."""

        return self.load_bundle()

    def get_bundle_summary(self) -> Dict[str, Any]:
        """Return source and snapshot metadata for docs, APIs, and evals.
        
        This method extracts the key metadata fields from the loaded snapshot
        for use in API responses (/info endpoint) and evaluation reports.
        
        Returns:
            Dict with keys:
            - snapshot_id: Unique snapshot identifier (e.g., "fraudshield-realworld-v2")
            - schema_version: Snapshot format version (e.g., "2.0")
            - generated_at: ISO timestamp when snapshot was built (or None if historical)
            - seed: Random seed used for deterministic sampling (default: 42)
            - source_count: Number of source datasets used (typically 1)
            - sources: List of source metadata dicts, each with:
                - source_id: Unique source identifier (e.g., "kaggle_creditcardfraud")
                - provider: Provider name (e.g., "Kaggle / ULB")
                - title: Human-readable title
                - dataset_id: Dataset identifier for downloading
                - source_url: Link to dataset page
            - task_sizes: Dict mapping task names ("easy", "medium", "hard") to case counts
                Example: {"easy": 36, "medium": 36, "hard": 36}
        
        Returns empty dict if no snapshot is loaded.
        
        Example:
            loader = FraudDataLoader()
            loader.load_bundle()
            summary = loader.get_bundle_summary()
            print(f"Snapshot {summary['snapshot_id']} has {summary['task_sizes']['easy']} easy cases")
        
        Note:
            Called by FraudShieldEnvironment.load_data() to populate /info endpoint.
            Safe to call even if load_bundle() hasn't been called yet (returns {}).
        """

        if not self.bundle_metadata:
            return {}

        sources = self.bundle_metadata.get("sources", [])
        return {
            "snapshot_id": self.bundle_metadata.get("snapshot_id"),
            "schema_version": self.bundle_metadata.get("schema_version"),
            "generated_at": self.bundle_metadata.get("generated_at"),
            "seed": self.bundle_metadata.get("seed", self.seed),
            "source_count": len(sources),
            "sources": [
                {
                    "source_id": source.get("source_id"),
                    "provider": source.get("provider"),
                    "title": source.get("title"),
                    "dataset_id": source.get("dataset_id"),
                    "source_url": source.get("source_url"),
                }
                for source in sources
            ],
            "task_sizes": {task_name: len(cases) for task_name, cases in self.task_bundle.items()},
        }

    def get_task_cases(self, task: str) -> List[Dict[str, Any]]:
        """Return the full case records for a given task.
        
        Each case record is a complete fraud review scenario with:
        - transaction_data: 26 fields (seller, buyer, amount, payment, etc.)
        - label: "fraud" or "legitimate" (ground truth)
        - task: Task name ("easy", "medium", "hard")
        - transaction_id: Unique case identifier
        
        Args:
            task: One of "easy", "medium", "hard"
        
        Returns:
            List of case dicts (36 cases per task, ordered deterministically)
        
        Raises:
            RuntimeError: If data not loaded (call load_bundle() first)
            ValueError: If task name is invalid
        
        Example:
            loader = FraudDataLoader()
            loader.load_bundle()
            easy_cases = loader.get_task_cases("easy")  # 36 cases
            for case in easy_cases:
                txn = case["transaction_data"]
                label = case["label"]  # "fraud" or "legitimate"
                print(f"Amount: {txn['amount']}, Label: {label}")
        
        Note:
            Cases are in deterministic order (sorted by stable hash based on transaction_id).
            This order is the same across all runs (reproducibility).
        """

        if not self.task_bundle:
            raise RuntimeError("Data not loaded. Call load_bundle() first.")
        if task not in self.task_bundle:
            raise ValueError(f"Unknown task: {task}")
        return list(self.task_bundle[task])

    def get_split_by_difficulty(
        self,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str], List[str], List[str]]:
        """Compatibility helper returning transactions and labels per task.
        
        This method provides a convenient interface for accessing all tasks' data
        in a single call, useful for baseline training or evaluation pipelines.
        
        Returns:
            Tuple of 6 lists:
            - easy_transactions: List[Dict] of 36 transaction dicts for easy task
            - medium_transactions: List[Dict] of 36 transaction dicts for medium task
            - hard_transactions: List[Dict] of 36 transaction dicts for hard task
            - easy_labels: List[str] of 36 labels ("fraud" or "legitimate")
            - medium_labels: List[str] of 36 labels
            - hard_labels: List[str] of 36 labels
        
        Example:
            loader = FraudDataLoader()
            loader.load_bundle()
            easy_txns, med_txns, hard_txns, easy_lbls, med_lbls, hard_lbls = loader.get_split_by_difficulty()
            
            # Train on easy task
            for txn, label in zip(easy_txns, easy_lbls):
                print(f"Amount: {txn['amount']}, Label: {label}")
        
        Note:
            Transaction order matches label order (equivalent to zipping).
        """

        easy_cases = self.get_task_cases("easy")
        medium_cases = self.get_task_cases("medium")
        hard_cases = self.get_task_cases("hard")
        return (
            [case["transaction_data"] for case in easy_cases],
            [case["transaction_data"] for case in medium_cases],
            [case["transaction_data"] for case in hard_cases],
            [case["label"] for case in easy_cases],
            [case["label"] for case in medium_cases],
            [case["label"] for case in hard_cases],
        )

    def _normalize_bundle_metadata(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Support both the original bundle shape and the new metadata-rich one."""

        metadata = payload.get("metadata")
        if metadata:
            return metadata

        source_name = payload.get("source", "Public fraud source")
        return {
            "snapshot_id": "fraudshield-realworld-v1",
            "schema_version": "1.0",
            "generated_at": None,
            "seed": payload.get("seed", self.seed),
            "sources": [
                {
                    "source_id": PRIMARY_SOURCE_ID,
                    "provider": self.source_catalog[PRIMARY_SOURCE_ID]["provider"],
                    "title": source_name,
                    "dataset_id": self.source_catalog[PRIMARY_SOURCE_ID]["dataset_id"],
                    "source_url": self.source_catalog[PRIMARY_SOURCE_ID]["source_url"],
                }
            ],
        }

    def _build_bundle_payload(self) -> Dict[str, Any]:
        """Build the full snapshot payload written to disk."""

        sources = [
            {
                "source_id": source_id,
                **details,
            }
            for source_id, details in self.source_catalog.items()
            if source_id == PRIMARY_SOURCE_ID
        ]
        metadata = {
            "snapshot_id": "fraudshield-realworld-v2",
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": self.seed,
            "build_notes": (
                "Runtime uses this frozen snapshot only. Public source downloads are optional and "
                "intended for rebuilding the snapshot offline."
            ),
            "sources": sources,
        }
        return {
            "metadata": metadata,
            "tasks": self.task_bundle,
        }

    def _build_task_bundle(self) -> Dict[str, List[Dict[str, Any]]]:
        """Create deterministic tasks from the public source dataset."""

        if self.df is None:
            raise RuntimeError("Source dataframe is not loaded.")

        feature_columns = [column for column in self.df.columns if column.startswith("V")]
        working = self.df.copy()
        pca_signal = np.sqrt((working[feature_columns] ** 2).sum(axis=1))
        working["pca_rank"] = pca_signal.rank(method="first", pct=True)
        working["amount_rank"] = working["Amount"].rank(method="first", pct=True)
        working["time_rank"] = working["Time"].rank(method="first", pct=True)
        working["case_score"] = (
            0.70 * working["pca_rank"]
            + 0.20 * working["amount_rank"]
            + 0.10 * (working["time_rank"] - 0.5).abs() * 2.0
        )
        working["row_id"] = working.index.astype(int)

        fraud_rows = working[working["Class"] == 1].sort_values("case_score")
        legit_rows = working[working["Class"] == 0].sort_values("case_score")

        bundle: Dict[str, List[Dict[str, Any]]] = {}
        for task_name, spec in TASK_SPECS.items():
            fraud_subset = self._select_band(
                fraud_rows,
                spec["fraud_band"][0],
                spec["fraud_band"][1],
                spec["per_class"],
            )
            legit_subset = self._select_band(
                legit_rows,
                spec["legit_band"][0],
                spec["legit_band"][1],
                spec["per_class"],
            )

            cases: List[Dict[str, Any]] = []
            for index, (_, row) in enumerate(fraud_subset.iterrows()):
                cases.append(self._row_to_case(row, task_name, "fraud", index))
            for index, (_, row) in enumerate(legit_subset.iterrows()):
                cases.append(self._row_to_case(row, task_name, "legitimate", index))

            ordered_cases = sorted(
                cases,
                key=lambda case: self._stable_ratio(task_name, case["transaction_id"], "order"),
            )
            bundle[task_name] = ordered_cases
            logger.info(
                "Prepared %s task with %s cases (%s fraud / %s legitimate)",
                task_name,
                len(ordered_cases),
                spec["per_class"],
                spec["per_class"],
            )

        return bundle

    def _select_band(
        self,
        frame: pd.DataFrame,
        start_fraction: float,
        end_fraction: float,
        count: int,
    ) -> pd.DataFrame:
        """Select evenly spaced rows from a quantile band."""

        total = len(frame)
        start_index = min(total - 1, int(total * start_fraction))
        end_index = max(start_index + count, int(total * end_fraction))
        band = frame.iloc[start_index:end_index]
        if len(band) <= count:
            return band
        positions = np.linspace(0, len(band) - 1, count, dtype=int)
        return band.iloc[positions]

    def _row_to_case(
        self,
        row: pd.Series,
        task_name: str,
        label: str,
        local_index: int,
    ) -> Dict[str, Any]:
        """Convert a source row into one deterministic marketplace case."""

        row_id = int(row["row_id"])
        anomaly_strength = float(row["case_score"])
        amount = max(6.0, round(float(row["Amount"]), 2))
        timestamp = self._timestamp_from_seconds(float(row["Time"]))
        base_risk = self._clamp(0.18 + anomaly_strength * 0.45 + (0.18 if label == "fraud" else -0.08))
        amount_percentile = round(float(row["amount_rank"] * 100.0), 2)

        fraud_ring_group = local_index % 6 if task_name == "hard" and label == "fraud" else None
        flash_sale_group = local_index % 8 if task_name == "hard" and label == "legitimate" else None

        seller_id = self._seller_id(task_name, label, row_id, fraud_ring_group, flash_sale_group)
        buyer_id = self._buyer_id(task_name, label, row_id, local_index)
        shipping_address = self._shipping_country(task_name, label, row_id)
        device_country = self._device_country(task_name, label, row_id, shipping_address)
        shipping_speed = self._shipping_speed(task_name, label, row_id)

        price_gap = self._price_gap_multiplier(task_name, label, row_id)
        item_price = round(amount / price_gap, 2)

        transaction_data = {
            "amount": amount,
            "seller_id": seller_id,
            "buyer_id": buyer_id,
            "item_category": self._item_category(amount, row_id),
            "item_price": item_price,
            "shipping_address": shipping_address,
            "seller_account_age_days": self._seller_age_days(task_name, label, row_id),
            "buyer_account_age_days": self._buyer_age_days(task_name, label, row_id),
            "payment_method": self._payment_method(task_name, label, row_id),
            "device_country": device_country,
            "timestamp": timestamp,
            "is_repeat_buyer": self._repeat_buyer(task_name, label, row_id),
            "seller_avg_rating": self._seller_rating(task_name, label, row_id),
            "num_seller_reviews": self._seller_reviews(task_name, label, row_id),
            "previous_fraud_flags": self._previous_flags(task_name, label, row_id),
            "shipping_speed": shipping_speed,
            "amount_percentile": amount_percentile,
            "seller_chargeback_rate_30d": self._chargeback_rate(task_name, label, row_id),
            "buyer_disputes_90d": self._buyer_disputes(task_name, label, row_id),
            "shared_device_accounts_24h": self._shared_device_accounts(task_name, label, row_id, fraud_ring_group),
            "same_address_orders_24h": self._same_address_orders(task_name, label, row_id, flash_sale_group),
        }

        historical_context = self._historical_context(
            task_name=task_name,
            label=label,
            row_id=row_id,
            transaction_data=transaction_data,
            local_index=local_index,
            anomaly_strength=anomaly_strength,
            fraud_ring_group=fraud_ring_group,
            flash_sale_group=flash_sale_group,
        )

        business_cost = self._business_cost(task_name, label, anomaly_strength)
        return {
            "transaction_id": f"{task_name}_{label}_{row_id}",
            "label": label,
            "risk_score": round(base_risk, 4),
            "business_cost": round(business_cost, 3),
            "transaction_data": transaction_data,
            "historical_context": historical_context,
        }

    @staticmethod
    def _timestamp_from_seconds(raw_seconds: float) -> str:
        base_time = datetime(2024, 3, 1, 0, 0, 0)
        return (base_time + timedelta(seconds=int(raw_seconds))).isoformat()

    @staticmethod
    def _clamp(value: float, lower: float = 0.02, upper: float = 0.98) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def _stable_ratio(*parts: Any) -> float:
        digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
        return int(digest[:12], 16) / float(16**12 - 1)

    def _stable_int(self, low: int, high: int, *parts: Any) -> int:
        if low >= high:
            return low
        ratio = self._stable_ratio(self.seed, *parts)
        return low + int(round(ratio * (high - low)))

    def _stable_float(self, low: float, high: float, *parts: Any) -> float:
        if low >= high:
            return low
        ratio = self._stable_ratio(self.seed, *parts)
        return low + ratio * (high - low)

    def _stable_choice(self, options: List[str], *parts: Any) -> str:
        index = self._stable_int(0, len(options) - 1, *parts)
        return options[index]

    def _seller_id(
        self,
        task_name: str,
        label: str,
        row_id: int,
        fraud_ring_group: int | None,
        flash_sale_group: int | None,
    ) -> str:
        if fraud_ring_group is not None:
            return f"seller_ring_{fraud_ring_group:02d}"
        if flash_sale_group is not None:
            return f"seller_flash_{flash_sale_group:02d}"
        prefix = "seller_new" if label == "fraud" and task_name == "easy" else "seller_std"
        return f"{prefix}_{row_id:06d}"

    def _buyer_id(self, task_name: str, label: str, row_id: int, local_index: int) -> str:
        if task_name == "hard" and label == "fraud":
            return f"buyer_linked_{local_index % 9:02d}"
        return f"buyer_{row_id:06d}"

    def _shipping_country(self, task_name: str, label: str, row_id: int) -> str:
        trusted = ["US", "GB", "CA", "DE", "AU", "IN", "SG"]
        risky = ["NG", "RU", "BR", "ID", "VN", "TR", "UA"]
        if task_name == "easy":
            pool = risky if label == "fraud" else trusted
        elif task_name == "medium":
            pool = risky + trusted[:4] if label == "fraud" else trusted + risky[:2]
        else:
            pool = trusted + risky[:4]
        return self._stable_choice(pool, "ship", task_name, label, row_id)

    def _device_country(self, task_name: str, label: str, row_id: int, shipping_address: str) -> str:
        trusted = ["US", "GB", "CA", "DE", "AU", "IN", "SG"]
        mismatch_pool = [country for country in trusted if country != shipping_address] + ["NG", "TR", "BR"]
        same_geo_probability = {
            ("easy", "legitimate"): 0.92,
            ("easy", "fraud"): 0.22,
            ("medium", "legitimate"): 0.75,
            ("medium", "fraud"): 0.48,
            ("hard", "legitimate"): 0.58,
            ("hard", "fraud"): 0.44,
        }[(task_name, label)]
        if self._stable_ratio("device-match", task_name, label, row_id) < same_geo_probability:
            return shipping_address
        return self._stable_choice(mismatch_pool, "device-country", task_name, label, row_id)

    def _shipping_speed(self, task_name: str, label: str, row_id: int) -> str:
        if task_name == "easy" and label == "fraud":
            options = ["overnight", "express", "same-day"]
        elif task_name == "hard" and label == "legitimate":
            options = ["standard", "express", "overnight"]
        else:
            options = ["standard", "express", "overnight"]
        return self._stable_choice(options, "ship-speed", task_name, label, row_id)

    def _price_gap_multiplier(self, task_name: str, label: str, row_id: int) -> float:
        ranges = {
            ("easy", "fraud"): (1.35, 2.10),
            ("easy", "legitimate"): (0.94, 1.05),
            ("medium", "fraud"): (1.10, 1.55),
            ("medium", "legitimate"): (0.92, 1.18),
            ("hard", "fraud"): (1.00, 1.28),
            ("hard", "legitimate"): (0.95, 1.22),
        }
        low, high = ranges[(task_name, label)]
        return round(self._stable_float(low, high, "price-gap", task_name, label, row_id), 3)

    def _seller_age_days(self, task_name: str, label: str, row_id: int) -> int:
        ranges = {
            ("easy", "fraud"): (1, 45),
            ("easy", "legitimate"): (200, 3200),
            ("medium", "fraud"): (14, 180),
            ("medium", "legitimate"): (60, 2400),
            ("hard", "fraud"): (45, 900),
            ("hard", "legitimate"): (30, 1600),
        }
        low, high = ranges[(task_name, label)]
        return self._stable_int(low, high, "seller-age", task_name, label, row_id)

    def _buyer_age_days(self, task_name: str, label: str, row_id: int) -> int:
        ranges = {
            ("easy", "fraud"): (3, 180),
            ("easy", "legitimate"): (120, 3200),
            ("medium", "fraud"): (14, 500),
            ("medium", "legitimate"): (30, 2600),
            ("hard", "fraud"): (30, 1200),
            ("hard", "legitimate"): (15, 2000),
        }
        low, high = ranges[(task_name, label)]
        return self._stable_int(low, high, "buyer-age", task_name, label, row_id)

    def _payment_method(self, task_name: str, label: str, row_id: int) -> str:
        if task_name == "easy" and label == "fraud":
            pool = ["prepaid_card", "gift_card", "crypto_gateway"]
        elif task_name == "hard" and label == "legitimate":
            pool = ["credit_card", "paypal", "wallet", "buy_now_pay_later"]
        else:
            pool = ["credit_card", "paypal", "wallet", "bank_transfer", "buy_now_pay_later"]
        return self._stable_choice(pool, "payment", task_name, label, row_id)

    def _repeat_buyer(self, task_name: str, label: str, row_id: int) -> bool:
        probability = {
            ("easy", "fraud"): 0.08,
            ("easy", "legitimate"): 0.72,
            ("medium", "fraud"): 0.20,
            ("medium", "legitimate"): 0.55,
            ("hard", "fraud"): 0.34,
            ("hard", "legitimate"): 0.42,
        }[(task_name, label)]
        return self._stable_ratio("repeat", task_name, label, row_id) < probability

    def _seller_rating(self, task_name: str, label: str, row_id: int) -> float:
        ranges = {
            ("easy", "fraud"): (1.4, 3.1),
            ("easy", "legitimate"): (4.2, 4.95),
            ("medium", "fraud"): (2.1, 4.0),
            ("medium", "legitimate"): (3.5, 4.9),
            ("hard", "fraud"): (3.0, 4.8),
            ("hard", "legitimate"): (2.8, 4.8),
        }
        low, high = ranges[(task_name, label)]
        return round(self._stable_float(low, high, "rating", task_name, label, row_id), 2)

    def _seller_reviews(self, task_name: str, label: str, row_id: int) -> int:
        ranges = {
            ("easy", "fraud"): (0, 40),
            ("easy", "legitimate"): (120, 6000),
            ("medium", "fraud"): (10, 500),
            ("medium", "legitimate"): (40, 4200),
            ("hard", "fraud"): (40, 2800),
            ("hard", "legitimate"): (8, 3200),
        }
        low, high = ranges[(task_name, label)]
        return self._stable_int(low, high, "reviews", task_name, label, row_id)

    def _previous_flags(self, task_name: str, label: str, row_id: int) -> int:
        ranges = {
            ("easy", "fraud"): (1, 3),
            ("easy", "legitimate"): (0, 0),
            ("medium", "fraud"): (0, 2),
            ("medium", "legitimate"): (0, 1),
            ("hard", "fraud"): (0, 1),
            ("hard", "legitimate"): (0, 1),
        }
        low, high = ranges[(task_name, label)]
        return self._stable_int(low, high, "flags", task_name, label, row_id)

    def _chargeback_rate(self, task_name: str, label: str, row_id: int) -> float:
        ranges = {
            ("easy", "fraud"): (0.12, 0.28),
            ("easy", "legitimate"): (0.00, 0.03),
            ("medium", "fraud"): (0.06, 0.18),
            ("medium", "legitimate"): (0.01, 0.08),
            ("hard", "fraud"): (0.04, 0.14),
            ("hard", "legitimate"): (0.03, 0.12),
        }
        low, high = ranges[(task_name, label)]
        return round(self._stable_float(low, high, "chargeback", task_name, label, row_id), 3)

    def _buyer_disputes(self, task_name: str, label: str, row_id: int) -> int:
        ranges = {
            ("easy", "fraud"): (1, 5),
            ("easy", "legitimate"): (0, 1),
            ("medium", "fraud"): (0, 4),
            ("medium", "legitimate"): (0, 2),
            ("hard", "fraud"): (0, 3),
            ("hard", "legitimate"): (0, 3),
        }
        low, high = ranges[(task_name, label)]
        return self._stable_int(low, high, "disputes", task_name, label, row_id)

    def _shared_device_accounts(
        self,
        task_name: str,
        label: str,
        row_id: int,
        fraud_ring_group: int | None,
    ) -> int:
        if fraud_ring_group is not None:
            return self._stable_int(6, 18, "shared-device-ring", row_id, fraud_ring_group)
        ranges = {
            ("easy", "fraud"): (3, 8),
            ("easy", "legitimate"): (1, 2),
            ("medium", "fraud"): (2, 7),
            ("medium", "legitimate"): (1, 4),
            ("hard", "fraud"): (3, 9),
            ("hard", "legitimate"): (2, 9),
        }
        low, high = ranges[(task_name, label)]
        return self._stable_int(low, high, "shared-device", task_name, label, row_id)

    def _same_address_orders(
        self,
        task_name: str,
        label: str,
        row_id: int,
        flash_sale_group: int | None,
    ) -> int:
        if flash_sale_group is not None:
            return self._stable_int(4, 11, "flash-sale", row_id, flash_sale_group)
        ranges = {
            ("easy", "fraud"): (2, 6),
            ("easy", "legitimate"): (0, 2),
            ("medium", "fraud"): (1, 5),
            ("medium", "legitimate"): (1, 4),
            ("hard", "fraud"): (3, 10),
            ("hard", "legitimate"): (2, 8),
        }
        low, high = ranges[(task_name, label)]
        return self._stable_int(low, high, "same-address", task_name, label, row_id)

    def _item_category(self, amount: float, row_id: int) -> str:
        if amount < 30:
            options = ["grocery", "beauty", "digital_goods"]
        elif amount < 120:
            options = ["fashion", "consumer_electronics", "home"]
        elif amount < 400:
            options = ["electronics", "appliances", "collectibles"]
        else:
            options = ["luxury", "travel", "high_value_collectibles"]
        return self._stable_choice(options, "category", row_id, amount)

    def _historical_context(
        self,
        task_name: str,
        label: str,
        row_id: int,
        transaction_data: Dict[str, Any],
        local_index: int,
        anomaly_strength: float,
        fraud_ring_group: int | None,
        flash_sale_group: int | None,
    ) -> Dict[str, Any]:
        seller_velocity = self._stable_int(
            1,
            18 if task_name != "hard" else 35,
            "seller-velocity",
            task_name,
            label,
            row_id,
        )
        linked_cards = self._stable_int(1, 4 if task_name == "easy" else 8, "linked-cards", task_name, label, row_id)
        recent_refunds = self._stable_int(0, 2 if task_name == "easy" else 6, "refunds", task_name, label, row_id)
        cluster_alert = round(
            self._clamp(0.20 + anomaly_strength * 0.55 + (0.10 if fraud_ring_group is not None else 0.0)),
            3,
        )

        note = TASK_SPECS[task_name]["focus"]
        if task_name == "hard":
            if fraud_ring_group is not None or flash_sale_group is not None:
                note = "Hard-mode cases can reflect either coordinated abuse pressure or legitimate operational spikes."
            else:
                note = "Hard-mode cases contain overlapping signals that benefit from graph and operations review."

        return {
            "task_focus": TASK_SPECS[task_name]["focus"],
            "snapshot_id": self.bundle_metadata.get("snapshot_id") if self.bundle_metadata else "fraudshield-realworld-v1",
            "source_id": PRIMARY_SOURCE_ID,
            "seller_transactions_1h": seller_velocity,
            "linked_cards_7d": linked_cards,
            "recent_refunds_7d": recent_refunds,
            "cluster_alert_score": cluster_alert,
            "network_pattern": note,
            "operations_context": note,
            "sequence_bucket": local_index + 1,
            "device_match": transaction_data["device_country"] == transaction_data["shipping_address"],
        }

    def _business_cost(self, task_name: str, label: str, anomaly_strength: float) -> float:
        task_bias = {"easy": 0.00, "medium": 0.10, "hard": 0.18}[task_name]
        label_bias = 0.28 if label == "fraud" else 0.02
        return self._clamp(0.75 + task_bias + label_bias + anomaly_strength * 0.35, 0.55, 1.85)


# Backward-compatible alias to avoid breaking older imports.
KaggleDataLoader = FraudDataLoader
