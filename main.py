from crewai import Crew, Process

from tasks import (
    create_campaign_tasks,
    create_action_proposal_task,
    create_adaptation_proposal_task,
)

from agents import (
    research_agent,
    copywriter_agent,
    art_director_agent,
    manager_agent,
)

import os
import json
import re
import time

from datetime import (
    datetime,
    timezone,
)

from dotenv import load_dotenv

from llm_config import get_llm

from models import (
    MerchantCampaignInput,
    ProposedCommerceAction,
    CampaignAdaptationProposal,
)

import policy_engine
import measurement
import merchant_analyzer


# =============================================================
# LOAD ENVIRONMENT
# =============================================================

load_dotenv()


# =============================================================
# COST CONFIGURATION
# =============================================================

PRICE_PER_1M_INPUT_TOKENS = 0.15
PRICE_PER_1M_OUTPUT_TOKENS = 0.60

RUN_LOG_FILE = "run_history.jsonl"


# =============================================================
# RETRY WAIT PARSER
# =============================================================

def _extract_retry_wait_seconds(
    exc: Exception,
    default: float = 30.0
) -> float:
    """
    Groq's rate-limit error message includes exactly how long
    to wait, for example:

    'Please try again in 29.37s'

    Parse that value instead of guessing a fixed delay.
    """

    match = re.search(
        r"try again in ([\d.]+)s",
        str(exc)
    )

    if match:

        try:

            return (
                float(match.group(1))
                + 2.0
            )

        except ValueError:

            pass

    return default


# =============================================================
# JSON CLEANING
# =============================================================

def _strip_markdown_fences(
    text: str
) -> str:
    """
    Remove markdown JSON fences if the model returns:

    ```json
    {...}
    ```

    even though it was instructed to return raw JSON.
    """

    text = text.strip()

    if text.startswith("```"):

        text = re.sub(
            r"^```[a-zA-Z]*\n?",
            "",
            text
        )

        text = re.sub(
            r"\n?```$",
            "",
            text
        )

        text = text.strip()

    return text


# =============================================================
# PARSE COMMERCE ACTION
# =============================================================

def _parse_proposed_action(
    raw_text: str
) -> ProposedCommerceAction:
    """
    Parse the Manager's raw JSON output into
    ProposedCommerceAction.
    """

    return (
        ProposedCommerceAction
        .model_validate_json(
            _strip_markdown_fences(
                raw_text
            )
        )
    )


# =============================================================
# PARSE CAMPAIGN #2 PROPOSAL
# =============================================================

def _parse_adaptation_proposal(
    raw_text: str
) -> CampaignAdaptationProposal:
    """
    Parse the Campaign #2 proposal JSON output.
    """

    return (
        CampaignAdaptationProposal
        .model_validate_json(
            _strip_markdown_fences(
                raw_text
            )
        )
    )


# =============================================================
# GET LATEST POLICY DECISION
# =============================================================

def _get_latest_policy_decision(
    campaign_id: str
) -> dict:
    """
    Read data/policy_log.jsonl and return the most recent
    decision for the campaign.

    Used only for the final campaign summary.

    Does NOT modify policy_engine.py.
    """

    path = os.path.join(
        "data",
        "policy_log.jsonl"
    )

    if not os.path.exists(path):

        return None

    latest = None

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:

                continue

            try:

                entry = json.loads(
                    line
                )

            except json.JSONDecodeError:

                continue

            if (
                entry.get("campaign_id")
                == campaign_id
            ):

                if (

                    latest is None

                    or

                    entry["logged_at"]
                    >
                    latest["logged_at"]

                ):

                    latest = entry

    return latest


# =============================================================
# RUN LOGGING
# =============================================================

def _log_run(
    campaign: MerchantCampaignInput,
    duration_s: float,
    usage,
    succeeded: bool
) -> dict:

    input_cost = (

        usage.prompt_tokens
        /
        1_000_000

    ) * PRICE_PER_1M_INPUT_TOKENS

    output_cost = (

        usage.completion_tokens
        /
        1_000_000

    ) * PRICE_PER_1M_OUTPUT_TOKENS


    record = {

        "timestamp":

            datetime.now(
                timezone.utc
            ).isoformat(),


        "merchant_name":

            campaign.merchant_name,


        "business_goal":

            campaign.business_goal,


        "product":

            campaign.product,


        "target_customers":

            campaign.target_customers,


        "campaign_budget":

            campaign.campaign_budget,


        "target_conversion":

            campaign.target_conversion,


        "succeeded":

            succeeded,


        "duration_seconds":

            round(
                duration_s,
                2
            ),


        "total_tokens":

            usage.total_tokens,


        "prompt_tokens":

            usage.prompt_tokens,


        "completion_tokens":

            usage.completion_tokens,


        "successful_requests":

            usage.successful_requests,


        "estimated_cost_usd":

            round(
                input_cost
                +
                output_cost,
                6
            ),
    }


    with open(
        RUN_LOG_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(

            json.dumps(
                record
            )

            +

            "\n"

        )


    return record


# =============================================================
# MAIN CAMPAIGN CREATOR
# =============================================================

def run_campaign_creator(
    campaign: MerchantCampaignInput
):

    """
    Run the multi-agent workflow for creating
    a merchant marketing campaign.
    """


    # =========================================================
    # START
    # =========================================================

    print(
        f"\n{'=' * 60}"
    )

    print(
        "Starting Marketing Campaign Creation"
    )

    print(
        campaign.as_prompt_block()
    )

    print(
        f"{'=' * 60}\n"
    )


    # =========================================================
    # MERCHANT DATA ANALYSIS
    # =========================================================

    merchant_metrics = (

        merchant_analyzer
        .analyze_merchant_data()

    )


    # =========================================================
    # CREATE CAMPAIGN TASKS
    # =========================================================

    tasks = (

        create_campaign_tasks(

            campaign,

            merchant_metrics,

        )

    )


    crew = Crew(

        agents=[

            research_agent,

            copywriter_agent,

            art_director_agent,

            manager_agent,

        ],

        tasks=tasks,

        process=Process.sequential,

        verbose=True,

    )


    # =========================================================
    # RUN MULTI-AGENT CAMPAIGN CREW
    # =========================================================

    start_time = time.monotonic()


    try:

        result = crew.kickoff()

        succeeded = True


    finally:

        duration_s = (

            time.monotonic()

            -

            start_time

        )


        usage = crew.usage_metrics


        if usage is not None:


            record = _log_run(

                campaign,

                duration_s,

                usage,

                locals().get(

                    "succeeded",

                    False,

                ),

            )


            print(

                f"\n📊 Run stats: "

                f"{record['total_tokens']} tokens, "

                f"${record['estimated_cost_usd']:.6f}, "

                f"{record['duration_seconds']}s "

                f"— logged to {RUN_LOG_FILE}"

            )


    # =========================================================
    # FINAL MARKETING BRIEF
    # =========================================================

    print(
        f"\n{'=' * 60}"
    )

    print(
        "FINAL MARKETING CAMPAIGN BRIEF"
    )

    print(
        f"{'=' * 60}\n"
    )


    brief_text = str(
        result
    )


    print(
        brief_text
    )


    filename = (

        f"campaign_brief_"

        f"{campaign.slug()}.txt"

    )


    with open(

        filename,

        "w",

        encoding="utf-8"

    ) as f:

        f.write(
            brief_text
        )


    print(

        f"\n✅ Brief saved to: "

        f"{filename}"

    )


    # =========================================================
    # STEP 3
    # AI PROPOSAL + POLICY REVIEW
    # =========================================================

    print(
        f"\n{'=' * 60}"
    )

    print(
        "PROPOSED COMMERCE ACTION"
    )

    print(
        f"{'=' * 60}\n"
    )


    max_attempts = 2


    for attempt in range(

        1,

        max_attempts + 1

    ):


        try:


            action_task = (

                create_action_proposal_task(

                    campaign,

                    brief_text

                )

            )


            action_crew = Crew(

                agents=[

                    manager_agent

                ],

                tasks=[

                    action_task

                ],

                process=Process.sequential,

                verbose=True,

            )


            action_result = (

                action_crew.kickoff()

            )


            proposed_action = (

                _parse_proposed_action(

                    str(
                        action_result
                    )

                )

            )


            # -------------------------------------------------
            # AUTHORITATIVE PRODUCT OVERRIDE
            # -------------------------------------------------

            proposed_action.product = (

                campaign.product

            )


            # -------------------------------------------------
            # AI PROPOSAL
            # -------------------------------------------------

            print(
                "\n[AI PROPOSAL]"
            )


            print(

                f"Proposal: "

                f"{proposed_action.model_dump()}\n"

            )


            # -------------------------------------------------
            # POLICY REVIEW
            # -------------------------------------------------

            print(
                "[POLICY REVIEW]"
            )


            policy_result = (

                policy_engine
                .process_proposed_action(

                    campaign,

                    proposed_action

                )

            )


            print(

                f"Status: "

                f"{policy_result['status']}"

            )


            print(

                f"Action ID: "

                f"{policy_result['action_id']}"

            )


            if policy_result["reasons"]:


                print(
                    "Reasons:"
                )


                for reason in (

                    policy_result[
                        "reasons"
                    ]

                ):


                    print(

                        f"  - {reason}"

                    )


            # -------------------------------------------------
            # EXECUTION / IDEMPOTENCY CHECK
            # -------------------------------------------------

            print(

                "\n[EXECUTION / IDEMPOTENCY CHECK]"

            )


            # -------------------------------------------------
            # NEW EXECUTION
            # -------------------------------------------------

            if (

                policy_result["status"]

                ==

                "APPROVED_AND_EXECUTED"

            ):


                executor = (

                    policy_result[
                        "execution"
                    ].get(

                        "executor",

                        "unknown"

                    )

                )


                print(

                    "Current run: "

                    "APPROVED_AND_EXECUTED"

                )


                print(

                    "A NEW action was created "

                    "in this run."

                )


                print(

                    f"Executor used: "

                    f"{executor}"

                )


                print(

                    f"Execution record: "

                    f"{policy_result['execution']}"

                )


            # -------------------------------------------------
            # DUPLICATE
            # -------------------------------------------------

            elif (

                policy_result["status"]

                ==

                "DUPLICATE_SKIPPED"

            ):


                prev = (

                    policy_result.get(

                        "previous_execution",

                        {}

                    )

                )


                print(

                    "Current run: "

                    "DUPLICATE_SKIPPED"

                )


                print(

                    "A previously executed action "

                    "was found."

                )


                print(

                    "No new Razorpay order/action "

                    "was created in this run."

                )


                if prev:


                    print(

                        f"Previously executed via: "

                        f"{prev.get('executor', 'unknown')}"

                    )


                    if (

                        prev.get("executor")

                        ==

                        "razorpay_test_mode"

                    ):


                        print(

                            "Previous Razorpay order ID: "

                            f"{prev.get('razorpay_order_id')}"

                        )


                    print(

                        "Originally executed at: "

                        f"{prev.get('executed_at')}"

                    )


            # -------------------------------------------------
            # EXECUTION FAILURE
            # -------------------------------------------------

            elif (

                policy_result["status"]

                ==

                "EXECUTION_FAILED"

            ):


                print(

                    "Current run: "

                    "EXECUTION_FAILED"

                )


                print(

                    "No action was created "

                    "in this run."

                )


                print(

                    "Executor attempted: "

                    f"{policy_result.get('executor_attempted', 'unknown')}"

                )


                print(

                    "This action was NOT "

                    "automatically retried."

                )


                print(

                    "It was also NOT written "

                    "to the idempotency store."

                )


            # -------------------------------------------------
            # REJECTED
            # -------------------------------------------------

            else:


                print(

                    "Current run: "

                    "REJECTED"

                )


                print(

                    "No action was created "

                    "in this run."

                )


            break


        except Exception as e:


            if attempt < max_attempts:


                wait_s = (

                    _extract_retry_wait_seconds(
                        e
                    )

                )


                print(

                    f"⚠️ Commerce action proposal "

                    f"attempt {attempt} failed ({e}). "

                    f"Waiting {wait_s:.0f}s "

                    f"then retrying once..."

                )


                time.sleep(
                    wait_s
                )


            else:


                print(

                    f"⚠️ Commerce action proposal "

                    f"failed after {attempt} attempts "

                    f"(brief was still generated "

                    f"and saved successfully): {e}"

                )


    return result


# =============================================================
# CLI ENTRY POINT
# =============================================================

if __name__ == "__main__":


    # =========================================================
    # LLM CONFIGURATION CHECK
    # =========================================================

    try:


        get_llm()


    except EnvironmentError as e:


        print(
            f"\n❌ {e}"
        )


        raise SystemExit(
            1
        )


    # =========================================================
    # CAMPAIGN INPUT
    # =========================================================

    print(

        "Enter the merchant campaign details "

        "(press Enter to use the example value):\n"

    )


    defaults = {


        "merchant_name":

            "UrbanFit Store",


        "business_goal":

            "Increase repeat purchases",


        "product":

            "Running Shoes",


        "target_customers":

            "Customers who have not purchased "

            "in the last 60 days",


        "campaign_budget":

            "₹5,000",


        "target_conversion":

            "10%",


    }


    values = {}


    for field, default in defaults.items():


        label = (

            field

            .replace(
                "_",
                " "
            )

            .title()

        )


        entered = input(

            f"{label} [{default}]: "

        ).strip()


        values[field] = (

            entered

            or

            default

        )


    campaign = (

        MerchantCampaignInput(

            **values

        )

    )


    # =========================================================
    # RUN CAMPAIGN
    # =========================================================

    run_campaign_creator(
        campaign
    )


    # =========================================================
    # CAMPAIGN STATE FOR SUMMARY
    # =========================================================

    campaign_id = campaign.slug()


    policy_decision = (

        _get_latest_policy_decision(
            campaign_id
        )

    )


    result_record = None

    campaign2_generated = False


    # =========================================================
    # STEP 4
    # MEASUREMENT
    # =========================================================

    executed_actions_path = (

        os.path.join(

            "data",

            "executed_actions.json"

        )

    )


    matching_action = None


    if os.path.exists(
        executed_actions_path
    ):


        with open(

            executed_actions_path,

            "r",

            encoding="utf-8"

        ) as f:


            executed_actions = (

                json.load(
                    f
                )

            )


        for record in (

            executed_actions.values()

        ):


            if (

                record.get(
                    "campaign_id"
                )

                ==

                campaign_id

            ):


                if (

                    matching_action is None

                    or

                    record["executed_at"]

                    >

                    matching_action[
                        "executed_at"
                    ]

                ):


                    matching_action = (
                        record
                    )


    # =========================================================
    # ONLY MEASURE IF AN ACTION EXISTS
    # =========================================================

    if matching_action is not None:


        print(
            f"\n{'=' * 60}"
        )


        print(

            "[MEASUREMENT] "

            "STEP 4: RECORD CAMPAIGN RESULT"

        )


        print(
            f"{'=' * 60}\n"
        )


        print(

            "An action was executed for this campaign."

        )


        print(

            f"Action ID: "

            f"{matching_action['action_id']}"

        )


        want_to_record = input(

            "\nRecord a real/actual result "

            "for this campaign now? (y/N): "

        ).strip().lower()


        # =====================================================
        # RECORD RESULT
        # =====================================================

        if want_to_record == "y":


            print(

                "\nEnter campaign performance data."

            )


            # -------------------------------------------------
            # PERFORMANCE DATA
            # -------------------------------------------------

            actual_conversion_str = input(

                "Actual conversion % achieved "

                "(e.g. 7): "

            ).strip()


            orders_str = input(

                "Number of orders generated: "

            ).strip()


            payments_str = input(

                "Number of successful payments: "

            ).strip()


            # -------------------------------------------------
            # REVENUE DATA
            # -------------------------------------------------

            print(

                "\nEnter revenue and campaign cost data."

            )


            baseline_revenue_str = input(

                "Baseline revenue before campaign (₹): "

            ).strip()


            actual_revenue_str = input(

                "Actual revenue after campaign (₹): "

            ).strip()


            discount_cost_str = input(

                "Total discount cost (₹): "

            ).strip()


            marketing_cost_str = input(

                "Marketing / campaign cost (₹): "

            ).strip()


            try:


                # =============================================
                # SAVE MEASUREMENT + CALCULATE ROI
                # =============================================

                result_record = (

                    measurement
                    .record_campaign_result(

                        campaign_id=campaign_id,

                        target_conversion=(

                            campaign.target_conversion

                        ),

                        actual_conversion_percent=float(

                            actual_conversion_str

                        ),

                        orders=int(
                            orders_str
                        ),

                        successful_payments=int(

                            payments_str

                        ),

                        baseline_revenue=float(

                            baseline_revenue_str

                        ),

                        actual_revenue=float(

                            actual_revenue_str

                        ),

                        discount_cost=float(

                            discount_cost_str

                        ),

                        marketing_cost=float(

                            marketing_cost_str

                        ),

                    )

                )


                # =============================================
                # DISPLAY RESULTS
                # =============================================

                print(
                    f"\n{'=' * 60}"
                )


                print(
                    "CAMPAIGN PERFORMANCE RESULT"
                )


                print(
                    f"{'=' * 60}"
                )


                print(

                    f"Conversion outcome: "

                    f"{result_record['outcome']}"

                )


                print(

                    f"Target conversion: "

                    f"{result_record['target_conversion_percent']}%"

                )


                print(

                    f"Actual conversion: "

                    f"{result_record['actual_conversion_percent']}%"

                )


                print(

                    f"\nOrders: "

                    f"{result_record['orders']}"

                )


                print(

                    f"Successful payments: "

                    f"{result_record['successful_payments']}"

                )


                # ---------------------------------------------
                # REVENUE
                # ---------------------------------------------

                print(

                    f"\nBaseline revenue: "

                    f"₹{result_record['baseline_revenue']:,.2f}"

                )


                print(

                    f"Actual revenue: "

                    f"₹{result_record['actual_revenue']:,.2f}"

                )


                print(

                    f"Incremental revenue: "

                    f"₹{result_record['incremental_revenue']:,.2f}"

                )


                # ---------------------------------------------
                # COST
                # ---------------------------------------------

                print(

                    f"\nDiscount cost: "

                    f"₹{result_record['discount_cost']:,.2f}"

                )


                print(

                    f"Marketing cost: "

                    f"₹{result_record['marketing_cost']:,.2f}"

                )


                print(

                    f"Total campaign cost: "

                    f"₹{result_record['total_campaign_cost']:,.2f}"

                )


                # ---------------------------------------------
                # FINAL BUSINESS IMPACT
                # ---------------------------------------------

                print(

                    f"\nNet gain: "

                    f"₹{result_record['net_gain']:,.2f}"

                )


                print(

                    f"ROI: "

                    f"{result_record['roi_percent']:.2f}%"

                )


                print(
                    f"{'=' * 60}\n"
                )


            except ValueError as e:


                print(

                    f"⚠️ Result not recorded: {e}"

                )


        else:


            print(
                "Skipped recording a result."
            )


        # =====================================================
        # STEP 5
        # FEEDBACK LOOP
        # =====================================================

        if result_record is not None:


            print(
                f"\n{'=' * 60}"
            )


            print(

                "[FEEDBACK LOOP] "

                "STEP 5: PROPOSE CAMPAIGN #2"

            )


            print(

                "Advisory only — "

                "nothing auto-executes."

            )


            print(
                f"{'=' * 60}\n"
            )


            want_adaptation = input(

                "Generate a Campaign #2 proposal "

                "based on this result? (y/N): "

            ).strip().lower()


            if want_adaptation == "y":


                brief_filename = (

                    f"campaign_brief_"

                    f"{campaign.slug()}.txt"

                )


                try:


                    with open(

                        brief_filename,

                        "r",

                        encoding="utf-8"

                    ) as f:


                        brief_text_for_adaptation = (

                            f.read()

                        )


                except FileNotFoundError:


                    brief_text_for_adaptation = (

                        "(brief file not found — "

                        "proceeding without it)"

                    )


                max_attempts = 2


                for attempt in range(

                    1,

                    max_attempts + 1

                ):


                    try:


                        # =====================================
                        # REAL MERCHANT DATA FOR CAMPAIGN #2
                        # =====================================

                        # Recompute merchant metrics here because
                        # run_campaign_creator() does not return
                        # merchant_metrics.
                        #
                        # This makes real eligible-customer data
                        # available to the Campaign #2 proposal.

                        merchant_metrics_for_adaptation = (

                            merchant_analyzer
                            .analyze_merchant_data()

                        )


                        # =====================================
                        # CREATE CAMPAIGN #2 PROPOSAL TASK
                        # =====================================

                        adaptation_task = (

                            create_adaptation_proposal_task(

                                campaign,

                                brief_text_for_adaptation,

                                matching_action,

                                result_record,

                                merchant_metrics_for_adaptation,

                            )

                        )


                        adaptation_crew = Crew(

                            agents=[

                                manager_agent

                            ],

                            tasks=[

                                adaptation_task

                            ],

                            process=Process.sequential,

                            verbose=True,

                        )


                        adaptation_result = (

                            adaptation_crew.kickoff()

                        )


                        proposal = (

                            _parse_adaptation_proposal(

                                str(
                                    adaptation_result
                                )

                            )

                        )


                        # =====================================
                        # CAMPAIGN #2 OUTPUT
                        # =====================================

                        print(
                            f"\n{'=' * 60}"
                        )


                        print(
                            "PROPOSED CAMPAIGN #2"
                        )


                        print(

                            "Review only — "

                            "not launched automatically"

                        )


                        print(
                            f"{'=' * 60}\n"
                        )


                        print(

                            f"Business Goal: "

                            f"{proposal.revised_business_goal}"

                        )


                        print(

                            f"Product: "

                            f"{proposal.revised_product}"

                        )


                        print(

                            f"Target Customers: "

                            f"{proposal.revised_target_customers}"

                        )


                        print(

                            f"Campaign Budget: "

                            f"{proposal.revised_campaign_budget}"

                        )


                        print(

                            f"Target Conversion: "

                            f"{proposal.revised_target_conversion}"

                        )


                        print(

                            "\nWhat changed and why: "

                            f"{proposal.changes_summary}"

                        )


                        print(

                            "\nTo actually run this, "

                            "start 'python main.py' again "

                            "and enter these values manually."

                        )


                        campaign2_generated = True


                        break


                    except Exception as e:


                        if attempt < max_attempts:


                            wait_s = (

                                _extract_retry_wait_seconds(
                                    e
                                )

                            )


                            print(

                                f"⚠️ Campaign #2 proposal "

                                f"attempt {attempt} failed ({e}). "

                                f"Waiting {wait_s:.0f}s, "

                                f"then retrying once..."

                            )


                            time.sleep(
                                wait_s
                            )


                        else:


                            print(

                                f"⚠️ Campaign #2 proposal "

                                f"failed after {attempt} "

                                f"attempts: {e}"

                            )


            else:


                print(

                    "Skipped generating "

                    "a Campaign #2 proposal."

                )


# =============================================================
# FINAL CAMPAIGN SUMMARY
# =============================================================

    if policy_decision is not None:


        policy_status = (

            policy_decision[
                "status"
            ]

        )


        if (

            policy_status

            ==

            "APPROVED_AND_EXECUTED"

        ):


            execution_note = (

                "Newly executed this run"

            )


        elif (

            policy_status

            ==

            "DUPLICATE_SKIPPED"

        ):


            execution_note = (

                "Skipped as duplicate "

                "(no new action created)"

            )


        else:


            execution_note = (

                "No action executed"

            )


    else:


        policy_status = (

            "N/A "

            "(no policy decision logged "

            "for this campaign)"

        )


        execution_note = (

            "No action executed"

        )


    print(
        f"\n{'=' * 60}"
    )


    print(
        "FINAL CAMPAIGN SUMMARY"
    )


    print(
        f"{'=' * 60}"
    )


    print(

        f"Campaign ID: "

        f"{campaign_id}"

    )


    print(

        f"Policy status: "

        f"{policy_status}"

    )


    print(

        f"Execution: "

        f"{execution_note}"

    )


    print(

        f"Measurement recorded: "

        f"{'Yes' if result_record is not None else 'No'}"

    )


    # =========================================================
    # ROI SUMMARY
    # =========================================================

    if result_record is not None:


        print(

            f"Incremental revenue: "

            f"₹{result_record['incremental_revenue']:,.2f}"

        )


        print(

            f"Total campaign cost: "

            f"₹{result_record['total_campaign_cost']:,.2f}"

        )


        print(

            f"Net gain: "

            f"₹{result_record['net_gain']:,.2f}"

        )


        print(

            f"ROI: "

            f"{result_record['roi_percent']:.2f}%"

        )


    print(

        f"Campaign #2 generated: "

        f"{'Yes' if campaign2_generated else 'No'}"

    )


    print(
        f"{'=' * 60}\n"
    )