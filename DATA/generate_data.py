import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sku_data(
    sku_ids=["SKU_001", "SKU_002", "SKU_003", "SKU_004"],
    start_date="2022-01-01",
    end_date="2024-03-31",
    seed=42
):
    np.random.seed(seed)
    random.seed(seed)
    records = []

    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Inject realistic events
    events = {
        "2022-10-24": ("Diwali Sale", 3.5),
        "2023-01-26": ("Republic Day Sale", 1.8),
        "2022-11-25": ("Black Friday", 2.2),
        "2023-03-08": ("Womens Day", 1.6),
        "2023-07-15": ("Price Drop", 2.0),
        "2023-10-14": ("Competitor OOS", 2.8),
        "2023-12-25": ("Christmas", 2.1),
        "2022-08-15": ("Independence Day", 1.5),
        "2023-04-14": ("Supply Disruption", 0.2),
        "2022-06-21": ("Flash Flood News", 0.3),
    }

    sku_profiles = {
        "SKU_001": {"base": 120, "category": "Electronics", "price": 4999},
        "SKU_002": {"base": 85,  "category": "Apparel",     "price": 1299},
        "SKU_003": {"base": 200, "category": "Grocery",     "price": 349},
        "SKU_004": {"base": 60,  "category": "Home Decor",  "price": 2199},
    }

    for sku in sku_ids:
        profile = sku_profiles[sku]
        base_demand = profile["base"]

        for date in date_range:
            date_str = str(date.date())
            noise = np.random.normal(0, base_demand * 0.1)

            # Weekly seasonality
            weekday_effect = 1.2 if date.weekday() >= 5 else 1.0

            # Yearly trend
            trend = 1 + (date - pd.Timestamp(start_date)).days / 1000 * 0.15

            demand = base_demand * weekday_effect * trend + noise

            # Apply events
            if date_str in events:
                event_name, multiplier = events[date_str]
                demand *= multiplier

            # Halo effect for 3 days post-event
            for event_date_str, (_, mult) in events.items():
                event_date = pd.Timestamp(event_date_str)
                diff = (date - event_date).days
                if 1 <= diff <= 3:
                    demand *= (1 + (mult - 1) * 0.3)

            price = profile["price"] * np.random.uniform(0.9, 1.1)
            is_promo = 1 if (date_str in events and "Sale" in events[date_str][0]) else 0

            records.append({
                "date": date.date(),
                "sku_id": sku,
                "category": profile["category"],
                "demand": max(0, round(demand)),
                "price": round(price, 2),
                "is_promotion": is_promo,
                "stock_available": np.random.randint(50, 500),
                "competitor_oos": 1 if date_str == "2023-10-14" else 0
            })

    df = pd.DataFrame(records)
    df.to_csv("data/sku_demand.csv", index=False)
    print(f"✅ Generated {len(df)} rows for {len(sku_ids)} SKUs")
    return df

if __name__ == "__main__":
    generate_sku_data()