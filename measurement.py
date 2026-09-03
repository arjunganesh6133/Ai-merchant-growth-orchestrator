"""
STEP 4: campaign measurement.

Records actual campaign outcomes (target vs. actual conversion, orders,
successful payments) against a campaign_id.

Revenue and ROI metrics are calculated deterministically:

    incremental_revenue = actual_revenue - baseline_revenue
    total_campaign_cost = discount_cost + marketing_cost
    net_gain = incremental_revenue - total_campaign_cost

    ROI (%) = net_gain / total_campaign_cost * 100

There is deliberately NO LLM involvement in measurement calculations.
All metrics must come from a real source, Razorpay Test Mode data,
merchant data, or manual input standing in for a real result.
"""

import json
import os
from datetime import datetime, timezone

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_RESULTS_PATH = os.path.join(_DATA_DIR, "campaign_results.json")


def _load_results() -> dict:
    if not os.path.exists(_RESULTS_PATH):
        return {}

    try:
        with open(_RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_results(store: dict) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)

    with open(_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)


def _parse_percent(value: str) -> float:
    """Extracts a numeric percentage from a free-text field like '10%'."""
    digits = "".join(c for c in value if c.isdigit() or c == ".")

    try:
        return float(digits)
    except ValueError:
        return 0.0


def record_campaign_result(
    campaign_id: str,
    target_conversion: str,
    actual_conversion_percent: float,
    orders: int,
    successful_payments: int,
    baseline_revenue: float,
    actual_revenue: float,
    discount_cost: float,
    marketing_cost: float,
) -> dict:
    """
    Records campaign results and calculates deterministic business metrics.

    Revenue metrics:
        incremental_revenue = actual_revenue - baseline_revenue
        total_campaign_cost = discount_cost + marketing_cost
        net_gain = incremental_revenue - total_campaign_cost
        roi_percent = net_gain / total_campaign_cost * 100

    Returns the complete measurement record.
    """

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    if not 0 <= actual_conversion_percent <= 100:
        raise ValueError(
            "actual_conversion_percent must be between 0 and 100."
        )

    if orders < 0:
        raise ValueError(
            "orders must be greater than or equal to 0."
        )

    if successful_payments < 0:
        raise ValueError(
            "successful_payments must be greater than or equal to 0."
        )

    if successful_payments > orders:
        raise ValueError(
            "successful_payments cannot exceed orders."
        )

    if baseline_revenue < 0:
        raise ValueError(
            "baseline_revenue must be greater than or equal to 0."
        )

    if actual_revenue < 0:
        raise ValueError(
            "actual_revenue must be greater than or equal to 0."
        )

    if discount_cost < 0:
        raise ValueError(
            "discount_cost must be greater than or equal to 0."
        )

    if marketing_cost < 0:
        raise ValueError(
            "marketing_cost must be greater than or equal to 0."
        )

    # ---------------------------------------------------------
    # CONVERSION OUTCOME
    # ---------------------------------------------------------

    target_percent = _parse_percent(target_conversion)

    if actual_conversion_percent > target_percent:
        outcome = "EXCEEDED_TARGET"

    elif actual_conversion_percent == target_percent:
        outcome = "MET_TARGET"

    else:
        outcome = "MISSED_TARGET"

    # ---------------------------------------------------------
    # DETERMINISTIC BUSINESS METRICS
    # ---------------------------------------------------------

    incremental_revenue = actual_revenue - baseline_revenue

    total_campaign_cost = discount_cost + marketing_cost

    net_gain = incremental_revenue - total_campaign_cost

    roi_percent = (
        (net_gain / total_campaign_cost) * 100
        if total_campaign_cost > 0
        else 0
    )

    # ---------------------------------------------------------
    # FINAL RECORD
    # ---------------------------------------------------------

    record = {
        "campaign_id": campaign_id,

        # Conversion metrics
        "target_conversion": target_conversion,
        "target_conversion_percent": target_percent,
        "actual_conversion_percent": actual_conversion_percent,
        "orders": orders,
        "successful_payments": successful_payments,
        "outcome": outcome,

        # Revenue metrics
        "baseline_revenue": baseline_revenue,
        "actual_revenue": actual_revenue,
        "incremental_revenue": incremental_revenue,

        # Campaign cost
        "discount_cost": discount_cost,
        "marketing_cost": marketing_cost,
        "total_campaign_cost": total_campaign_cost,

        # Final business impact
        "net_gain": net_gain,
        "roi_percent": round(roi_percent, 2),

        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }

    store = _load_results()

    # One current measurement record per campaign
    store[campaign_id] = record

    _save_results(store)

    return record


def get_campaign_result(campaign_id: str) -> dict:
    """
    Returns the recorded result for campaign_id,
    or None if no measurement exists.
    """
    return _load_results().get(campaign_id)