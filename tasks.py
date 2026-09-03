from crewai import Task

from agents import (
    research_agent,
    copywriter_agent,
    art_director_agent,
    manager_agent,
)

from models import MerchantCampaignInput


# =============================================================
# STEP 1 + STEP 2
# MULTI-AGENT CAMPAIGN CREATION
# =============================================================

def create_campaign_tasks(
    campaign: MerchantCampaignInput,
    merchant_metrics: dict,
):
    """
    Create all tasks for the marketing campaign.

    merchant_metrics is computed from real merchant data.
    The primary grounding mechanism for Campaign #1 remains
    the lookup_merchant_data tool used by the Research Agent.
    """

    context_block = campaign.as_prompt_block()


    # =========================================================
    # TASK 1: RESEARCH
    # =========================================================

    research_task = Task(

        description=f"""
        Research current trends and audience insights for the
        following merchant campaign:

        {context_block}

        First, use the lookup_merchant_data tool to pull REAL
        customer data matching this campaign's Target Customers
        description and Product.

        For example:

        - If Target Customers says
          "have not purchased in the last 60 days",
          call the tool with min_days_inactive=60.

        - Pass the product name too when relevant.

        Base your audience insights on what this tool actually
        returns.

        IMPORTANT GROUNDING RULES:

        - Do NOT invent demographics such as age, profession,
          income, job type, location, or interests unless they
          are explicitly present in the merchant data.

        - Do NOT claim a customer segment has a spending threshold
          unless the tool actually provides that information.

        - If the tool returns no matching customers, say so
          clearly instead of inventing a segment.

        - Treat the campaign input and merchant data returned by
          the tool as the authoritative source for customer facts.

        Then provide:

        1. Top 3-5 current trends relevant to this product
           and audience.

        2. Target audience preferences grounded in the actual
           customer data returned by the tool.

        3. Competitor analysis.

        4. Key messaging opportunities that support the stated
           business goal.
        """,

        agent=research_agent,

        expected_output="""
        Concise research report under 200 words containing:

        - Real customer segment insights from lookup_merchant_data.
        - 3-5 relevant trends.
        - Audience preferences grounded in available data.
        - 1-2 competitor observations.
        - A clear messaging opportunity.

        Do not invent unsupported customer demographics or
        customer attributes.
        """,

    )


    # =========================================================
    # TASK 2: COPYWRITING
    # =========================================================

    copywriting_task = Task(

        description=f"""
        Based on the research findings, create compelling ad copy
        for the following merchant campaign:

        {context_block}

        Deliver:

        1. Main headline that is attention-grabbing.

        2. Sub-headline explaining the value proposition.

        3. Body copy of 3-4 sentences.

        4. Clear call-to-action.

        5. Three headline variations.

        IMPORTANT:

        Base customer messaging on the research findings and
        campaign input.

        Do not introduce unsupported customer demographics,
        spending levels, or customer attributes.
        """,

        agent=copywriter_agent,

        expected_output="""
        Ad copy under 150 words containing:

        - Main headline.
        - Sub-headline.
        - 3-4 sentence body copy.
        - CTA.
        - Three headline variations.
        """,

        context=[
            research_task
        ],

    )


    # =========================================================
    # TASK 3: ART DIRECTION
    # =========================================================

    art_direction_task = Task(

        description=f"""
        Create visual direction and image prompts for the following
        merchant campaign:

        {context_block}

        Use the campaign objective and research insights to guide
        the creative direction.

        Provide:

        1. Visual concept description.

        2. Color palette recommendations.

        3. Two to three detailed image prompts suitable for
           image-generation systems such as DALL-E or
           Stable Diffusion.

        4. Layout suggestions.

        Keep the visual direction relevant to the actual product
        and campaign audience.

        Do not invent demographic characteristics that are not
        supported by the campaign input or research.
        """,

        agent=art_director_agent,

        expected_output="""
        Visual direction containing:

        - Clear visual concept.
        - Color recommendations.
        - 2-3 detailed image generation prompts.
        - Layout recommendations.
        """,

        context=[
            research_task,
            copywriting_task,
        ],

    )


    # =========================================================
    # TASK 4: FINAL ASSEMBLY
    # =========================================================

    management_task = Task(

        description=f"""
        Compile all elements into a comprehensive marketing
        campaign brief for the following merchant campaign:

        {context_block}

        Assemble:

        1. Executive Summary.

        2. Research Insights.

        3. Complete Ad Copy including all variations.

        4. Visual Direction and Image Prompts.

        5. Implementation Recommendations.

        6. Success Metrics tied directly to the target conversion
           specified in the campaign input.

        IMPORTANT GROUNDING RULE:

        Preserve the factual customer insights from the research
        task.

        Do not introduce new customer demographics, spending
        thresholds, audience facts, or merchant data that were not
        present in the campaign input or research findings.

        Format the output as a professional marketing brief.
        """,

        agent=manager_agent,

        expected_output="""
        Complete professional marketing campaign brief with:

        - Executive summary.
        - Grounded research insights.
        - Complete ad copy.
        - Visual direction.
        - Implementation recommendations.
        - Success metrics.
        """,

        context=[
            research_task,
            copywriting_task,
            art_direction_task,
        ],

    )


    return [

        research_task,

        copywriting_task,

        art_direction_task,

        management_task,

    ]


# =============================================================
# STEP 3
# COMMERCE ACTION PROPOSAL
# =============================================================

def create_action_proposal_task(
    campaign: MerchantCampaignInput,
    brief_text: str,
) -> Task:

    """
    Build the structured commerce-action proposal task.

    This task runs in its own separate Crew so that a failure
    during action proposal cannot destroy the already completed
    marketing brief.

    The Manager only proposes an action.

    policy_engine.py is responsible for deterministic validation,
    authorization, and execution.
    """

    context_block = campaign.as_prompt_block()


    propose_action_task = Task(

        description=f"""
        Based on the campaign brief below, propose ONE concrete
        commerce action to help achieve this campaign's objective.

        Campaign input:

        {context_block}

        Campaign brief:

        {brief_text}

        Propose a single promotional action, such as a percentage
        discount on the campaign's product.

        Keep the action conservative and directly justified by
        the campaign brief.

        Do not propose a deeper discount than is reasonably needed
        to support the identified campaign objective.

        RULES FOR THE FIELDS:

        - product:
          Must be the EXACT product name from the campaign input.

        - discount_percent:
          Must be a reasonable number and must not exceed 20.

        - proposed_spend:
          Your estimated total ₹ cost of this action.

          It must not exceed the campaign's stated budget.

        - environment:
          Always set exactly to "test".

          This system only operates in a test or sandbox commerce
          environment.

          Never production.

        - rationale:
          Write 1-2 sentences connecting the proposed action to
          the campaign brief and target conversion.

        IMPORTANT:

        You are ONLY PROPOSING this action.

        You are NOT authorizing it.

        You are NOT executing it.

        A separate deterministic policy system will validate the
        proposal and decide whether it should be approved and
        executed.

        OUTPUT FORMAT:

        Respond with ONLY one raw JSON object.

        Do NOT include markdown code fences.

        Do NOT include explanations.

        Do NOT include any text before or after the JSON.

        The JSON must exactly match this structure:

        {{
          "product": "...",
          "discount_percent": 0,
          "proposed_spend": 0,
          "environment": "test",
          "rationale": "..."
        }}
        """,

        agent=manager_agent,

        expected_output="""
        A single raw JSON object with exactly these keys:

        product,
        discount_percent,
        proposed_spend,
        environment,
        rationale.

        No markdown.
        No additional text.
        """,

    )


    return propose_action_task


# =============================================================
# STEP 5
# CAMPAIGN #2 ADAPTATION / FEEDBACK LOOP
# =============================================================

def create_adaptation_proposal_task(

    campaign: MerchantCampaignInput,

    brief_text: str,

    executed_action: dict,

    result_record: dict,

    merchant_metrics: dict,

) -> Task:

    """
    STEP 5: Feedback loop.

    The Manager evaluates Campaign #1's real measured result and
    proposes a Campaign #2.

    merchant_metrics contains REAL merchant-data analysis.

    This is particularly important for Campaign #2 audience
    selection because the Manager must use the real eligible
    customer count instead of inventing audience thresholds.

    This task only PROPOSES Campaign #2.

    Nothing is automatically executed.
    """


    context_block = campaign.as_prompt_block()


    # =========================================================
    # EXTRACT REAL MERCHANT METRICS
    # =========================================================

    eligible_customer_count = (

        merchant_metrics.get(

            "campaign_2_eligible_customers_count",

            0,

        )

    )


    eligible_customer_ids = (

        merchant_metrics.get(

            "campaign_2_eligible_customer_ids",

            [],

        )

    )


    # Optional additional metrics.
    # These safely use .get() so the code will not crash if
    # merchant_analyzer.py does not provide them.

    inactive_30_90_count = (

        merchant_metrics.get(

            "inactive_30_90_days_count",

            None,

        )

    )


    footwear_60_180_count = (

        merchant_metrics.get(

            "footwear_60_180_days_count",

            None,

        )

    )


    # =========================================================
    # BUILD OPTIONAL DATA BLOCK
    # =========================================================

    additional_metrics_block = ""


    if inactive_30_90_count is not None:

        additional_metrics_block += f"""

        - Customers inactive between 30 and 90 days:
          {inactive_30_90_count}
        """


    if footwear_60_180_count is not None:

        additional_metrics_block += f"""

        - Customers with qualifying footwear purchase history:
          {footwear_60_180_count}
        """


    # =========================================================
    # CREATE CAMPAIGN #2 TASK
    # =========================================================

    adaptation_task = Task(

        description=f"""
        Campaign #1 has already run and its REAL result has been
        measured.

        Evaluate why Campaign #1 performed this way and propose
        what Campaign #2 should look like.

        ========================================================

        CAMPAIGN #1 ORIGINAL DEFINITION

        ========================================================

        {context_block}


        ========================================================

        CAMPAIGN #1 MARKETING BRIEF

        ========================================================

        {brief_text}


        ========================================================

        REAL COMMERCE ACTION EXECUTED FOR CAMPAIGN #1

        ========================================================

        This action is a REAL fact.

        It was already executed.

        Do NOT propose this exact action again as though it has
        not happened.

        - Product:
          {executed_action.get('product')}

        - Discount:
          {executed_action.get('discount_percent')}%

        - Spend:
          ₹{executed_action.get('proposed_spend')}

        - Environment:
          {executed_action.get('environment')}


        ========================================================

        REAL MEASURED RESULT FROM CAMPAIGN #1

        ========================================================

        These numbers are REAL recorded campaign results.

        They are NOT predictions.

        You MUST use these exact values when evaluating
        Campaign #1.

        Do NOT replace them with assumptions.

        Do NOT invent different numbers.

        - Target conversion:
          {result_record.get('target_conversion_percent')}%

        - Actual conversion achieved:
          {result_record.get('actual_conversion_percent')}%

        - Orders generated:
          {result_record.get('orders')}

        - Successful payments:
          {result_record.get('successful_payments')}

        - Baseline revenue:
          ₹{result_record.get('baseline_revenue')}

        - Actual revenue:
          ₹{result_record.get('actual_revenue')}

        - Incremental revenue:
          ₹{result_record.get('incremental_revenue')}

        - Total campaign cost:
          ₹{result_record.get('total_campaign_cost')}

        - Net gain:
          ₹{result_record.get('net_gain')}

        - ROI:
          {result_record.get('roi_percent')}%

        - Outcome:
          {result_record.get('outcome')}


        ========================================================

        REAL MERCHANT DATA FOR CAMPAIGN #2

        ========================================================

        The following information comes from deterministic
        analysis of the merchant's actual customer data.

        These are REAL data-grounded facts.

        Campaign #2 eligible customers:

        - Eligible customer count:
          {eligible_customer_count}

        - Eligible customer IDs:
          {eligible_customer_ids}

        {additional_metrics_block}


        IMPORTANT DATA GROUNDING RULES:

        The eligible customer count above is the authoritative
        number available for the Campaign #2 audience identified
        by the merchant data analysis.

        If you describe Campaign #2 as targeting this eligible
        segment, your description MUST be consistent with the
        real eligible customer count.

        Do NOT invent:

        - historical spending thresholds

        - income levels

        - age groups

        - professions

        - geographic locations

        - customer demographics

        - purchase values

        - customer attributes

        unless they are explicitly present in the Campaign #1
        input, Campaign #1 brief, or the real merchant data
        provided above.

        For example:

        WRONG:

        "Target customers with historical spending above ₹10,000"

        unless a ₹10,000 spending threshold was actually provided
        in the real merchant data.

        CORRECT:

        "Target the data-qualified eligible customer segment
        identified by the merchant analysis"

        or a more specific description that is directly supported
        by the real merchant data above.


        ========================================================

        YOUR TASK

        ========================================================

        Based on:

        1. Campaign #1's original definition.

        2. Campaign #1's marketing brief.

        3. The REAL action that was executed.

        4. The REAL measured campaign performance.

        5. The REAL merchant customer data provided above.

        Propose a revised Campaign #2.

        You may change:

        - Business goal.

        - Target customers.

        - Product.

        - Campaign budget.

        - Target conversion.

        You may also keep some values unchanged if the measured
        results suggest they were not the problem.


        ========================================================

        REASONING REQUIREMENT

        ========================================================

        Your changes_summary MUST explain the proposal using
        specific real numbers.

        For example:

        "Campaign #1 achieved 7% conversion against a 10% target,
        missing by 3 percentage points. Campaign #2 should adjust
        the targeting strategy..."

        Do NOT provide vague reasoning such as:

        "The campaign did not perform well, so we should improve
        the audience."

        Your explanation must directly reference the measured
        results and, where relevant, the real eligible audience
        data.


        ========================================================

        SAFETY / EXECUTION RULE

        ========================================================

        You are ONLY PROPOSING Campaign #2.

        Nothing will automatically execute.

        Nothing will automatically create a payment.

        Nothing will automatically create a Razorpay order.

        A human must review the proposal before any future campaign
        is launched.


        ========================================================

        OUTPUT FORMAT

        ========================================================

        Respond with ONLY a single raw JSON object.

        Do NOT include markdown code fences.

        Do NOT include explanations before the JSON.

        Do NOT include explanations after the JSON.

        Do NOT include extra text.

        The JSON must exactly follow this structure:

        {{
          "revised_business_goal": "...",
          "revised_product": "...",
          "revised_target_customers": "...",
          "revised_campaign_budget": "...",
          "revised_target_conversion": "...",
          "changes_summary": "..."
        }}
        """,

        agent=manager_agent,

        expected_output="""
        A single raw JSON object with exactly these keys:

        revised_business_goal,
        revised_product,
        revised_target_customers,
        revised_campaign_budget,
        revised_target_conversion,
        changes_summary.

        No markdown fences.

        No additional text.
        """,

    )


    return adaptation_task