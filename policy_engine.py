"""
STEP 3 + RAZORPAY TEST MODE: deterministic policy, authorization,
idempotency, and execution layer.

Core principle (from the project spec): "LLM proposes; deterministic software
authorizes." Nothing in this file calls an LLM. It takes a ProposedCommerceAction
(produced by the Manager Agent) and:

  1. VALIDATES it against fixed, hardcoded rules (max discount, budget ceiling,
     environment, product must exist in the merchant dataset).
  2. Generates a deterministic action_id from the campaign + action content, so
     re-running the exact same proposal always produces the exact same ID.
  3. Checks that action_id against a local idempotency log before executing —
     an identical action is never executed twice.
  4. EXECUTES only if both validation passed and it's not a duplicate, via
     ONE of two interchangeable executors sharing the same execution
     boundary:
       - LocalSafeExecutor (_execute_stub): no HTTP call, no real
         Razorpay contact. Used automatically when Razorpay test-mode
         credentials aren't configured, so the rest of the pipeline keeps
         working exactly as before without requiring Razorpay setup.
       - RazorpayTestExecutor (_execute_razorpay_test_mode): a real Orders
         API call (POST /v1/orders) against Razorpay's Test Mode — same
         code path as production, but test keys mean no real money moves.
         Used automatically when RAZORPAY_KEY_ID/RAZORPAY_KEY_SECRET are set.

  Satisfies the four required demo scenarios:
    TEST 1 (valid action)   -> policy APPROVED, executor runs, audit recorded
    TEST 2 (invalid action) -> REJECTED before any executor is even selected
    TEST 3 (duplicate)      -> DUPLICATE_SKIPPED before any executor runs
    TEST 4 (API failure)    -> caught, NOT retried automatically, recorded
                                as EXECUTION_FAILED, NOT written to the
                                idempotency store (since nothing succeeded)
"""

import hashlib
import json
import os
from datetime import datetime, timezone

import razorpay

from models import MerchantCampaignInput, ProposedCommerceAction

# ---- Fixed, deterministic policy rules ----
MAX_DISCOUNT_PERCENT = 20.0
ALLOWED_ENVIRONMENT = "test"

# ---- Razorpay Test Mode credentials (never hardcoded, env vars only) ----
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")


def _razorpay_configured() -> bool:
    """True only if both test-mode credentials are set. Mirrors the existing
    Gemini-fallback pattern in llm_config.py: without credentials, the
    system automatically falls back to the local stub — nothing breaks for
    anyone running this without Razorpay set up."""
    return bool(RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET)


def _get_razorpay_client() -> "razorpay.Client":
    return razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_MERCHANT_DATA_PATH = os.path.join(_DATA_DIR, "merchant_data.json")
_EXECUTED_ACTIONS_PATH = os.path.join(_DATA_DIR, "executed_actions.json")
_POLICY_LOG_PATH = os.path.join(_DATA_DIR, "policy_log.jsonl")


def _log_policy_decision(result: dict) -> None:
    """STEP 6: appends every policy decision (REJECTED, DUPLICATE_SKIPPED,
    APPROVED_AND_EXECUTED) to a log, so policy rejection rate and duplicate-
    prevention rate can be computed from real logged events later, instead
    of estimated or guessed at. Append-only, never read/rewritten by this
    module — a pure observer that doesn't affect validate/execute logic."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    entry = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "status": result["status"],
        "campaign_id": result["campaign_id"],
        "action_id": result["action_id"],
        "reasons": result.get("reasons", []),
    }
    with open(_POLICY_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _load_products() -> list:
    try:
        with open(_MERCHANT_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("products", [])
    except Exception:
        return []


def _load_executed_actions() -> dict:
    if not os.path.exists(_EXECUTED_ACTIONS_PATH):
        return {}
    try:
        with open(_EXECUTED_ACTIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_executed_actions(store: dict) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_EXECUTED_ACTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)


def _parse_budget(campaign_budget: str) -> float:
    """Extracts a numeric value from a free-text budget field like '₹5,000'."""
    digits = "".join(c for c in campaign_budget if c.isdigit() or c == ".")
    try:
        return float(digits)
    except ValueError:
        return 0.0


def generate_campaign_id(campaign: MerchantCampaignInput) -> str:
    """Deterministic campaign identifier, reusing the same slug logic as filenames."""
    return campaign.slug()


def generate_action_id(campaign_id: str, action: ProposedCommerceAction) -> str:
    """Deterministic action identifier derived from the campaign + the action's
    actual content. Two identical proposals for the same campaign always produce
    the same action_id (enabling idempotency); a genuinely different proposal
    (different discount, different spend) produces a different ID, since it's a
    genuinely different action, not a duplicate."""
    fingerprint = (
        f"{campaign_id}|{action.product.strip().lower()}|"
        f"{action.discount_percent}|{action.proposed_spend}"
    )
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:12]
    return f"ACT_{digest}"


def validate_action(campaign: MerchantCampaignInput, action: ProposedCommerceAction) -> tuple:
    """Returns (approved: bool, reasons: list[str]). Every check runs (doesn't
    short-circuit on the first failure) so a rejected proposal comes back with
    a complete list of everything wrong with it."""
    reasons = []

    if action.environment != ALLOWED_ENVIRONMENT:
        reasons.append(
            f"environment must be exactly '{ALLOWED_ENVIRONMENT}', got '{action.environment}'."
        )

    if action.discount_percent > MAX_DISCOUNT_PERCENT:
        reasons.append(
            f"discount_percent {action.discount_percent} exceeds the maximum "
            f"allowed of {MAX_DISCOUNT_PERCENT}%."
        )

    campaign_budget = _parse_budget(campaign.campaign_budget)
    if action.proposed_spend > campaign_budget:
        reasons.append(
            f"proposed_spend ₹{action.proposed_spend} exceeds the campaign budget "
            f"of ₹{campaign_budget} ({campaign.campaign_budget})."
        )

    products = _load_products()
    product_names = [p["product_name"].lower() for p in products]
    if action.product.strip().lower() not in product_names:
        reasons.append(
            f"product '{action.product}' was not found in the merchant dataset "
            f"(known products: {', '.join(p['product_name'] for p in products)})."
        )

    return (len(reasons) == 0, reasons)


def _execute_stub(campaign_id: str, action_id: str, action: ProposedCommerceAction) -> dict:
    """SAFE LOCAL STUB — no real payment/Razorpay API call. Just records that
    this action was executed. This is used automatically whenever Razorpay
    test-mode credentials aren't configured."""
    return {
        "campaign_id": campaign_id,
        "action_id": action_id,
        "product": action.product,
        "discount_percent": action.discount_percent,
        "proposed_spend": action.proposed_spend,
        "environment": action.environment,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "note": "TEST STUB — no real commerce/payment action was performed.",
        "executor": "local_stub",
    }


def _execute_razorpay_test_mode(campaign_id: str, action_id: str, action: ProposedCommerceAction) -> dict:
    """RAZORPAY TEST MODE EXECUTOR — a real Orders API call (POST /v1/orders)
    against Razorpay, using test-mode credentials. Same code path as
    production; test keys mean no real money ever moves (confirmed directly
    from Razorpay's own docs: test mode uses a mock bank page, no real
    settlement). Raises on failure — the caller (process_proposed_action)
    is responsible for catching it, recording EXECUTION_FAILED, and NOT
    retrying automatically (per spec rule: never blindly retry a financial
    action) and NOT writing it to the idempotency store, since nothing
    actually succeeded.

    Maps our fields onto Orders API's fields:
      - amount: proposed_spend converted to paise (Razorpay's currency
        subunit for INR), as an integer — Orders API requires subunits.
      - currency: INR (fixed — this project only ever proposes ₹ budgets).
      - receipt: action_id (our own deterministic ID, for our own
        cross-reference — Razorpay does not interpret this).
      - notes: campaign_id, product, discount_percent, environment — visible
        on the Razorpay dashboard for audit purposes.
    """
    client = _get_razorpay_client()

    order = client.order.create(
        data={
            "amount": int(round(action.proposed_spend * 100)),  # INR -> paise
            "currency": "INR",
            "receipt": action_id,
            "notes": {
                "campaign_id": campaign_id,
                "product": action.product,
                "discount_percent": str(action.discount_percent),
                "environment": action.environment,
            },
        }
    )

    return {
        "campaign_id": campaign_id,
        "action_id": action_id,
        "product": action.product,
        "discount_percent": action.discount_percent,
        "proposed_spend": action.proposed_spend,
        "environment": action.environment,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "executor": "razorpay_test_mode",
        "razorpay_order_id": order.get("id"),
        "razorpay_order_status": order.get("status"),
        "razorpay_amount_paise": order.get("amount"),
        "note": "REAL Razorpay Test Mode Orders API call — no real money moved (test keys).",
    }


def process_proposed_action(campaign: MerchantCampaignInput, action: ProposedCommerceAction) -> dict:
    """Runs the full validate -> idempotency-check -> execute pipeline.
    Returns a dict summary with a 'status' of one of:
      REJECTED                  — failed policy validation, not executed
      DUPLICATE_SKIPPED         — passed validation but action_id already executed before
      EXECUTION_FAILED          — passed validation, new action, but the executor
                                   (Razorpay or stub) raised — NOT retried automatically,
                                   NOT written to the idempotency store
      APPROVED_AND_EXECUTED     — passed validation, new action, executed successfully
    """
    campaign_id = generate_campaign_id(campaign)
    action_id = generate_action_id(campaign_id, action)

    approved, reasons = validate_action(campaign, action)
    if not approved:
        result = {
            "status": "REJECTED",
            "campaign_id": campaign_id,
            "action_id": action_id,
            "reasons": reasons,
        }
        _log_policy_decision(result)
        return result

    executed_actions = _load_executed_actions()
    if action_id in executed_actions:
        result = {
            "status": "DUPLICATE_SKIPPED",
            "campaign_id": campaign_id,
            "action_id": action_id,
            "reasons": ["This exact action was already executed previously."],
            "previous_execution": executed_actions[action_id],
        }
        _log_policy_decision(result)
        return result

    execution_record = None
    try:
        if _razorpay_configured():
            execution_record = _execute_razorpay_test_mode(campaign_id, action_id, action)
        else:
            execution_record = _execute_stub(campaign_id, action_id, action)
    except Exception as e:
        # TEST 4 (spec-required failure demo): the executor raised. Do NOT
        # retry automatically (never blindly retry a financial action), and
        # do NOT write anything to the idempotency store — nothing actually
        # succeeded, so a future legitimate retry attempt must not be
        # blocked by a phantom "already executed" entry.
        result = {
            "status": "EXECUTION_FAILED",
            "campaign_id": campaign_id,
            "action_id": action_id,
            "reasons": [f"Executor raised an error: {e}"],
            "executor_attempted": "razorpay_test_mode" if _razorpay_configured() else "local_stub",
        }
        _log_policy_decision(result)
        return result

    executed_actions = _load_executed_actions()
    executed_actions[action_id] = execution_record
    _save_executed_actions(executed_actions)

    result = {
        "status": "APPROVED_AND_EXECUTED",
        "campaign_id": campaign_id,
        "action_id": action_id,
        "reasons": [],
        "execution": execution_record,
    }
    _log_policy_decision(result)
    return result