"""Deterministic FraudShield task bundle loader."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "per_class": 12,
        "fraud_band": (0.70, 1.00),
        "legit_band": (0.00, 0.30),
        "focus": "Obvious fraud markers such as new sellers, price gaps, and geo mismatch.",
    },
    "medium": {
        "per_class": 18,
        "fraud_band": (0.35, 0.75),
        "legit_band": (0.25, 0.65),
        "focus": "Mixed-signal reviews where no single indicator is decisive.",
    },
    "hard": {
        "per_class": 24,
        "fraud_band": (0.00, 0.45),
        "legit_band": (0.60, 1.00),
        "focus": "Coordinated abuse and high-volume legitimate edge cases with overlap.",
    },
}


class KaggleDataLoader:
    """Loads the curated FraudShield task bundle or builds it from Kaggle data."""

    def __init__(self, data_path: str = "data", seed: int = 42):
        self.seed = seed
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)
        self.csv_file = self.data_path / "creditcard.csv"
        self.bundle_file = self.data_path / "fraudshield_cases.json"
        self.df: pd.DataFrame | None = None
        self.task_bundle: Dict[str, List[Dict[str, Any]]] = {}

    def download_data(self) -> bool:
        """Download the source Kaggle dataset for bundle regeneration."""

        try:
            import kaggle

            logger.info("Downloading Kaggle Credit Card Fraud dataset...")
            kaggle.api.dataset_download_files(
                "mlg-ulb/creditcardfraud",
                path=str(self.data_path),
                unzip=True,
            )
            logger.info("Downloaded source data to %s", self.data_path)
            return True
        except Exception as exc:  # pragma: no cover - external dependency
            logger.error("Failed to download Kaggle data: %s", exc)
            return False

    def load_data(self) -> bool:
        """Load the compact task bundle, building it from CSV if needed."""

        if self.bundle_file.exists():
            self.task_bundle = json.loads(self.bundle_file.read_text(encoding="utf-8"))["tasks"]
            logger.info("Loaded curated task bundle from %s", self.bundle_file)
            return True

        if not self.csv_file.exists():
            logger.error("Neither %s nor %s is available.", self.bundle_file, self.csv_file)
            return False

        logger.info("Building curated task bundle from %s", self.csv_file)
        self.df = pd.read_csv(self.csv_file)
        self.task_bundle = self._build_task_bundle()
        payload = {
            "seed": self.seed,
            "source": "Kaggle creditcardfraud",
            "tasks": self.task_bundle,
        }
        self.bundle_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote deterministic task bundle to %s", self.bundle_file)
        return True

    def get_task_cases(self, task: str) -> List[Dict[str, Any]]:
        """Return the full case records for a given task."""

        if not self.task_bundle:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        if task not in self.task_bundle:
            raise ValueError(f"Unknown task: {task}")
        return list(self.task_bundle[task])

    def get_split_by_difficulty(
        self,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str], List[str], List[str]]:
        """Compatibility helper returning transactions and labels per task."""

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

    def _build_task_bundle(self) -> Dict[str, List[Dict[str, Any]]]:
        """Create deterministic tasks from the Kaggle dataset."""

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
        """Convert a Kaggle row into one deterministic marketplace case."""

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
        case = {
            "transaction_id": f"{task_name}_{label}_{row_id}",
            "label": label,
            "risk_score": round(base_risk, 4),
            "business_cost": round(business_cost, 3),
            "transaction_data": transaction_data,
            "historical_context": historical_context,
        }
        return case

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
        seller_velocity = self._stable_int(1, 18 if task_name != "hard" else 35, "seller-velocity", task_name, label, row_id)
        linked_cards = self._stable_int(1, 4 if task_name == "easy" else 8, "linked-cards", task_name, label, row_id)
        recent_refunds = self._stable_int(0, 2 if task_name == "easy" else 6, "refunds", task_name, label, row_id)
        cluster_alert = round(self._clamp(0.20 + anomaly_strength * 0.55 + (0.10 if fraud_ring_group is not None else 0.0)), 3)

        note = TASK_SPECS[task_name]["focus"]
        if fraud_ring_group is not None:
            note = "Multiple hard-task cases reuse the same seller and device cluster."
        elif flash_sale_group is not None:
            note = "This seller is running a flash sale, so high order velocity may still be legitimate."

        return {
            "task_focus": TASK_SPECS[task_name]["focus"],
            "seller_transactions_1h": seller_velocity,
            "linked_cards_7d": linked_cards,
            "recent_refunds_7d": recent_refunds,
            "cluster_alert_score": cluster_alert,
            "network_pattern": note,
            "sequence_bucket": local_index + 1,
            "device_match": transaction_data["device_country"] == transaction_data["shipping_address"],
        }

    def _business_cost(self, task_name: str, label: str, anomaly_strength: float) -> float:
        task_bias = {"easy": 0.00, "medium": 0.10, "hard": 0.18}[task_name]
        label_bias = 0.28 if label == "fraud" else 0.02
        return self._clamp(0.75 + task_bias + label_bias + anomaly_strength * 0.35, 0.55, 1.85)
