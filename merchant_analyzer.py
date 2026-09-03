import json
import os
from datetime import datetime


DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "merchant_data.json"
)


def load_merchant_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_merchant_data():
    data = load_merchant_data()

    as_of_date = datetime.strptime(
        data["as_of_date"],
        "%Y-%m-%d"
    ).date()

    customers = data["customers"]
    products = data["products"]
    orders = data["orders"]

    # Product lookup
    product_map = {
        product["product_id"]: product
        for product in products
    }

    # 30–90 day inactive customers
    inactive_30_90 = []

    for customer in customers:
        last_purchase = datetime.strptime(
            customer["last_purchase"],
            "%Y-%m-%d"
        ).date()

        days_inactive = (
            as_of_date - last_purchase
        ).days

        if 30 <= days_inactive <= 90:
            inactive_30_90.append(
                customer["customer_id"]
            )

    # Customers who bought shoes in last 180 days
    # but not within last 60 days
    footwear_buyers = set()

    for order in orders:

        product = product_map.get(
            order["product_id"]
        )

        if not product:
            continue

        order_date = datetime.strptime(
            order["date"],
            "%Y-%m-%d"
        ).date()

        days_ago = (
            as_of_date - order_date
        ).days

        is_footwear = (
            "shoe" in product["product_name"].lower()
        )

        if (
            is_footwear
            and 60 < days_ago <= 180
            and order["payment_status"] == "paid"
        ):
            footwear_buyers.add(
                order["customer_id"]
            )

    # Combined audience
    eligible_customers = set(
        inactive_30_90
    ) | footwear_buyers

    return {
        "as_of_date": str(as_of_date),

        "total_customers":
            len(customers),

        "inactive_30_90_days_count":
            len(inactive_30_90),

        "inactive_30_90_customer_ids":
            inactive_30_90,

        "footwear_buyers_60_180_days_count":
            len(footwear_buyers),

        "footwear_buyer_customer_ids":
            sorted(footwear_buyers),

        "campaign_2_eligible_customers_count":
            len(eligible_customers),

        "campaign_2_eligible_customer_ids":
            sorted(eligible_customers)
    }