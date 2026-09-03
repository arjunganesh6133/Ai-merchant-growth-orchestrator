"""
Shared input model for the campaign workflow.

STEP 1 change: the crew used to take a single free-text `campaign_topic`
string. It now takes a structured merchant/business objective instead, so
the agents have a business goal, product, target audience, budget, and
target metric to reason about rather than a bare topic.

Defined once here so tasks.py, main.py, and api.py all validate/consume the
exact same shape instead of each re-declaring the fields.
"""

import re

from pydantic import BaseModel, Field, field_validator


class MerchantCampaignInput(BaseModel):
    """Structured business objective for a single campaign request."""

    merchant_name: str = Field(..., description="Name of the merchant/business, e.g. 'UrbanFit Store'")
    business_goal: str = Field(..., description="What the campaign should achieve, e.g. 'Increase repeat purchases'")
    product: str = Field(..., description="Product or product line the campaign is about, e.g. 'Running Shoes'")
    target_customers: str = Field(..., description="Who the campaign should target, e.g. 'Customers who have not purchased in the last 60 days'")
    campaign_budget: str = Field(..., description="Campaign budget, e.g. '₹5,000'")
    target_conversion: str = Field(..., description="Target conversion rate/metric, e.g. '10%'")

    @field_validator("target_customers")
    @classmethod
    def target_customers_must_be_descriptive(cls, v: str) -> str:
        """Catches the exact mistake of a bare number (e.g. '5000') landing in this
        field instead of a real description like 'inactive for 60+ days'. A bare
        number has no way to be turned into a real filter downstream, and silently
        gets treated as if it were a genuine (unfiltered) description — which then
        lets an LLM fabricate an audience size from it. Reject it here instead."""
        stripped = v.strip().replace(",", "")
        if stripped.isdigit():
            raise ValueError(
                f"target_customers must describe WHO to target (e.g. 'customers "
                f"inactive for 60+ days'), not a bare number like '{v}'. "
                f"A number alone can't be turned into an audience filter."
            )
        return v

    @field_validator("campaign_budget", "target_conversion")
    @classmethod
    def must_contain_a_number(cls, v: str, info) -> str:
        """Catches the reverse mistake: a description landing in a field that's
        supposed to hold a number (e.g. a budget or a percentage)."""
        if not re.search(r"\d", v):
            raise ValueError(
                f"{info.field_name} must contain a number (e.g. '₹5,000' or '10%'), got '{v}'."
            )
        return v

    def as_prompt_block(self) -> str:
        """Render as a labeled block for interpolation into task descriptions."""
        return (
            f"Merchant: {self.merchant_name}\n"
            f"Business Goal: {self.business_goal}\n"
            f"Product: {self.product}\n"
            f"Target Customers: {self.target_customers}\n"
            f"Campaign Budget: {self.campaign_budget}\n"
            f"Target Conversion: {self.target_conversion}"
        )

    def slug(self) -> str:
        """Filesystem/log-safe short identifier, used where campaign_topic used to be."""
        raw = f"{self.merchant_name}_{self.product}"
        return "".join(c if c.isalnum() else "_" for c in raw).strip("_")


class ProposedCommerceAction(BaseModel):
    """STEP 3: structured commerce action the Manager Agent proposes after
    assembling the campaign brief. Deliberately does NOT include campaign_id
    or action_id — those are assigned by policy_engine.py (deterministic code),
    not the LLM, so idempotency doesn't depend on an LLM being consistent
    across runs. This model is only the *proposal*; policy_engine.py is what
    actually validates and authorizes it. ("LLM proposes; deterministic
    software authorizes.")
    """

    product: str = Field(..., description="Exact product name this action applies to — must match the campaign's product.")
    discount_percent: float = Field(..., description="Proposed discount percentage, e.g. 15.0")
    proposed_spend: float = Field(..., description="Estimated total ₹ cost of this action, must not exceed the campaign budget.")
    environment: str = Field(..., description="Must always be 'test' — this system never proposes production actions.")
    rationale: str = Field(..., description="1-2 sentence justification tying this action to the research findings and target conversion goal.")

    @field_validator("discount_percent")
    @classmethod
    def discount_must_be_sane(cls, v: float) -> float:
        if v < 0 or v > 100:
            raise ValueError(f"discount_percent must be between 0 and 100, got {v}.")
        return v

    @field_validator("proposed_spend")
    @classmethod
    def spend_must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"proposed_spend cannot be negative, got {v}.")
        return v


class CampaignAdaptationProposal(BaseModel):
    """STEP 5 (feedback loop, propose-only): the Manager's suggestion for
    what "Campaign #2" should look like, given a real recorded result for
    Campaign #1. Purely advisory — nothing in the system auto-executes this;
    it's printed for the person to review and, if they choose, manually
    launch as a fresh campaign via main.py. Because nothing here gets
    executed or checked against real money/budget, this doesn't need
    policy_engine-style strict validation the way ProposedCommerceAction does.
    """

    revised_business_goal: str = Field(..., description="Business goal for Campaign #2 (may be unchanged from Campaign #1).")
    revised_product: str = Field(..., description="Product for Campaign #2 (may be unchanged).")
    revised_target_customers: str = Field(..., description="Target customer description for Campaign #2.")
    revised_campaign_budget: str = Field(..., description="Budget for Campaign #2, e.g. '₹5,000'.")
    revised_target_conversion: str = Field(..., description="Target conversion for Campaign #2, e.g. '10%'.")
    changes_summary: str = Field(..., description="What changed from Campaign #1 and why, grounded in the real recorded result — not invented.")