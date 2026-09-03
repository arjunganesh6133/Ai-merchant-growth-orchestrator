import json
from datetime import datetime

import os
import re
import time

import requests
from crewai.tools import tool
from tavily import TavilyClient


_MERCHANT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "merchant_data.json")


@tool
def lookup_merchant_data(min_days_inactive: int = 0, favourite_category: str = "", product_name: str = "") -> str:
    """Look up REAL merchant customer and product data from the local dataset.
    Use this instead of guessing or inventing demographics.

    Args:
        min_days_inactive: only include customers whose last purchase was at least
            this many days ago (relative to the dataset's as_of_date). Use 0 for no filter.
            Parse this from the campaign's "Target Customers" field, e.g. "have not
            purchased in the last 60 days" -> min_days_inactive=60.
        favourite_category: only include customers whose favourite_category matches
            (case-insensitive). Leave blank for no filter.
        product_name: if set, also returns matching product info (price, inventory)
            for the campaign's product. Leave blank to skip.

    Returns a plain-text summary of matching customers (id, days inactive, number of
    purchases, total spend, favourite category), aggregate stats, and product info if requested.
    If nothing matches, says so explicitly rather than fabricating results.
    """
    try:
        with open(_MERCHANT_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return f"ERROR: could not load merchant dataset — {e}"

    as_of = datetime.strptime(data["as_of_date"], "%Y-%m-%d")
    total_customers_in_dataset = len(data.get("customers", []))
    no_filters_applied = (min_days_inactive == 0 and not favourite_category)

    matches = []
    for c in data.get("customers", []):
        last_purchase = datetime.strptime(c["last_purchase"], "%Y-%m-%d")
        days_inactive = (as_of - last_purchase).days

        if days_inactive < min_days_inactive:
            continue
        if favourite_category and c["favourite_category"].lower() != favourite_category.lower():
            continue

        matches.append((c, days_inactive))

    lines = [f"Merchant customer data (as of {data['as_of_date']}):\n"]
    lines.append(
        f"AUTHORITATIVE FACT: this merchant's dataset has exactly "
        f"{total_customers_in_dataset} customers total. Any customer count or "
        f"audience-size figure you state anywhere in your output MUST come from "
        f"this tool's numbers — never from the campaign budget, conversion target, "
        f"or any other field."
    )

    if no_filters_applied:
        lines.append(
            f"\n⚠️ NO FILTERS APPLIED — the values below below are for the FULL, "
            f"UNFILTERED dataset ({total_customers_in_dataset} customers), NOT a "
            f"targeted segment. If the campaign's Target Customers field was meant "
            f"to describe a specific segment, re-check that it was parsed correctly "
            f"before treating this as your target audience."
        )

    if not matches:
        lines.append(
            f"\nNo customers matched the given filters "
            f"(min_days_inactive={min_days_inactive}, favourite_category='{favourite_category}')."
        )
    else:
        lines.append(f"\nMatching customers ({len(matches)} of {total_customers_in_dataset} total):")
        for c, days_inactive in matches:
            lines.append(
                f"- {c['customer_id']}: inactive {days_inactive} days, "
                f"{c['number_of_purchases']} previous purchases, "
                f"total spend ₹{c['total_spend']}, "
                f"favourite category: {c['favourite_category']}"
            )
        total_spend = sum(c["total_spend"] for c, _ in matches)
        avg_purchases = sum(c["number_of_purchases"] for c, _ in matches) / len(matches)
        lines.append(
            f"\nAggregate: {len(matches)} customers, "
            f"avg {avg_purchases:.1f} previous purchases each, "
            f"combined total spend ₹{total_spend}."
        )

    if product_name:
        product_match = next(
            (p for p in data.get("products", []) if p["product_name"].lower() == product_name.lower()),
            None,
        )
        if product_match:
            lines.append(
                f"\nProduct info for '{product_name}': "
                f"price ₹{product_match['price']}, "
                f"inventory {product_match['inventory']} units."
            )
        else:
            lines.append(f"\nNo product found in the dataset matching '{product_name}'.")

    return "\n".join(lines)


@tool
def research_trends(topic: str) -> str:
    """Research current trends for a given topic using a real web search."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return (
            "ERROR: TAVILY_API_KEY is not set. Get a free key at "
            "https://tavily.com and add it to your .env file."
        )

    client = TavilyClient(api_key=api_key)

    try:
        results = client.search(
            query=f"current trends {topic} 2026",
            search_depth="basic",
            max_results=3,
            include_answer=True,
        )
    except Exception as e:
        return f"ERROR: Tavily search failed — {e}"

    lines = [f"Research results for: {topic}\n"]

    if results.get("answer"):
        lines.append(f"Summary: {results['answer']}\n")

    lines.append("Sources:")
    for i, item in enumerate(results.get("results", []), start=1):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        content = (item.get("content") or "").strip().replace("\n", " ")
        snippet = content[:150] + ("..." if len(content) > 150 else "")
        lines.append(f"{i}. {title} ({url})\n   {snippet}")

    return "\n".join(lines)


@tool
def generate_image_prompt(description: str) -> str:
    """Generate a real marketing image from a description using Pollinations AI
    (free, no API key required) and save it locally."""
    full_prompt = (
        f"Professional marketing visual, {description}, high quality, "
        f"8k resolution, vibrant colors, modern aesthetic, clean composition, "
        f"commercial photography style"
    )

    encoded_prompt = requests.utils.quote(full_prompt)
    image_url = (
        f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        f"?width=1024&height=1024&nologo=true"
    )

    try:
        response = requests.get(image_url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        return f"ERROR: Image generation request failed — {e}"

    os.makedirs("generated_images", exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", description)[:50].strip("_")
    filename = f"generated_images/{safe_name}_{int(time.time())}.jpg"

    with open(filename, "wb") as f:
        f.write(response.content)

    return (
        f"Image generated successfully.\n"
        f"Prompt used: {full_prompt}\n"
        f"Saved to: {filename}\n"
        f"Source URL: {image_url}"
    )