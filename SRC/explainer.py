"""
explainer.py
Reads anomaly data from Excel, builds rich context prompts,
calls the Google Gemini API to generate natural-language root-cause reports.

FREE TIER: Gemini 1.5 Flash → 15 requests/min, 1M tokens/day — no billing needed.

Setup:
  1. Go to https://aistudio.google.com/app/apikey
  2. Click "Create API Key" (free, no credit card)
  3. Add to .env:  GEMINI_API_KEY=your_key_here
  4. pip install google-generativeai python-dotenv
"""

import google.genai as genai
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

from SRC.anomaly_detector import load_raw_data

SYSTEM_PROMPT = """You are a senior retail demand analyst with 15+ years of experience 
in FMCG, e-commerce, and consumer electronics. You specialize in diagnosing demand 
anomalies using statistical signals and business context.

Your job is to analyze SKU-level demand anomalies and generate clear, actionable 
root cause reports for operations and supply chain managers.

Always be specific, business-focused, and prioritize actionable insights.
Format your response in clean sections with emojis for readability."""


def build_context_prompt(anomaly_row: dict, sku_history: pd.DataFrame) -> str:
    """Build a rich context prompt for each anomaly."""
    date = anomaly_row.get("Date", anomaly_row.get("date", ""))
    sku = anomaly_row.get("SKU_ID", anomaly_row.get("sku_id", ""))
    category = anomaly_row.get("Category", anomaly_row.get("category", ""))
    actual = anomaly_row.get("Demand", anomaly_row.get("demand", 0))
    expected = round(anomaly_row.get("Forecast", anomaly_row.get("forecast", 0)), 1)
    deviation = anomaly_row.get("Deviation_Pct", anomaly_row.get("deviation_pct", 0))
    atype = anomaly_row.get("Anomaly_Type", anomaly_row.get("anomaly_type", "Normal")).capitalize()
    price = anomaly_row.get("Price", anomaly_row.get("price", 0))
    promo = anomaly_row.get("Is_Promotion", anomaly_row.get("is_promotion", 0))
    comp_oos = anomaly_row.get("Competitor_OOS", anomaly_row.get("competitor_oos", 0))
    high_conf = anomaly_row.get("High_Confidence", anomaly_row.get("high_confidence", False))
    iso_score = round(anomaly_row.get("ISO_Score", anomaly_row.get("iso_score", 0)), 3)

    # Get surrounding 7-day context
    df_context = sku_history.copy()
    df_context["Date"] = pd.to_datetime(df_context["Date"])
    target = pd.Timestamp(date)
    window = df_context[
        (df_context["Date"] >= target - pd.Timedelta(days=7)) &
        (df_context["Date"] <= target + pd.Timedelta(days=3))
    ][["Date", "Demand", "Price", "Is_Promotion"]].to_string(index=False)

    # Month-over-month average
    month = pd.Timestamp(date).month
    mom_avg = df_context[
        pd.to_datetime(df_context["Date"]).dt.month == month
    ]["Demand"].mean()
    if pd.isna(mom_avg):
        mom_avg = df_context["Demand"].mean()  # fallback to overall average

    prompt = f"""
Analyze the following demand anomaly and provide a structured root cause report.

━━━━━━━━━━━━━━━━━━━━━━
ANOMALY DETAILS
━━━━━━━━━━━━━━━━━━━━━━
- SKU ID         : {sku}
- Category       : {category}
- Date           : {date}
- Type           : {atype.upper()} anomaly
- Actual Demand  : {actual} units
- Expected Range : {round(anomaly_row.get("Forecast_Lower", anomaly_row.get("yhat_lower", 0)),1)} – {round(anomaly_row.get("Forecast_Upper", anomaly_row.get("yhat_upper", 0)),1)} units
- Prophet Forecast: {expected} units
- Deviation      : {deviation:+.1f}% from forecast
- Isolation Score: {iso_score} (closer to -1 = more anomalous)
- High Confidence: {"YES – flagged by both models" if high_conf else "Moderate – flagged by one model"}

━━━━━━━━━━━━━━━━━━━━━━
BUSINESS SIGNALS
━━━━━━━━━━━━━━━━━━━━━━
- Price on day   : ₹{price}
- Active Promo   : {"YES" if promo else "NO"}
- Competitor OOS : {"YES" if comp_oos else "NO"}
- Month avg demand (historical): {round(mom_avg, 1)} units

━━━━━━━━━━━━━━━━━━━━━━
7-DAY SURROUNDING CONTEXT
━━━━━━━━━━━━━━━━━━━━━━
{window}

━━━━━━━━━━━━━━━━━━━━━━
YOUR TASK
━━━━━━━━━━━━━━━━━━━━━━
Provide a structured report with these exact sections:

## 🔍 Root Cause Analysis
Identify the most likely cause(s) with confidence level (High/Medium/Low).
Consider: festivals, promotions, competitor activity, supply issues, 
seasonal patterns, price sensitivity, news events.

## 📊 Supporting Evidence
What signals in the data support your conclusion?

## ⚠️ Business Impact
Quantify the impact: revenue gain/loss, stockout risk, overstock risk.

## ✅ Recommended Actions
3–5 concrete, prioritized actions the operations or supply chain team should take NOW.
"""
    return prompt


def explain_anomaly(sku_id, date_str, path):
    """Call Gemini API to generate explanation for a single anomaly."""
    raw = load_raw_data(path)
    sku_history = raw[raw['SKU_ID'] == sku_id]
    anomaly_row = sku_history[sku_history['Date'] == pd.to_datetime(date_str)].iloc[0].to_dict()
    
    if not client:
        return "Error: GEMINI_API_KEY not set in .env file. Please add your Gemini API key."
    
    prompt = build_context_prompt(anomaly_row, sku_history)

    try:
        # Use the correct google.genai API - client.models.generate_content()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        error_str = str(e)
        # Try alternative model if primary fails
        if 'not found' in error_str.lower() or '404' in error_str:
            try:
                response = client.models.generate_content(
                    model="gemini-pro",
                    contents=prompt
                )
                return response.text
            except:
                pass
        
        return f"Error: Could not connect to Gemini API.\nMake sure:\n1. GEMINI_API_KEY is set in .env (get from https://aistudio.google.com/app/apikey)\n2. Your API key is valid\n3. You have quota remaining\n\nDetails: {error_str[:200]}"


def get_top_anomalies(n, excel_path):
    """Get top N anomalies by absolute deviation."""
    from SRC.anomaly_detector import load_anomaly_summary
    anom = load_anomaly_summary(excel_path)
    top = anom.reindex(anom['Deviation_Pct'].abs().nlargest(n).index)
    return top


def batch_explain(
    n: int = 5,
    excel_path: str = "SKU_Anomaly_Dashboard_PowerBI.xlsx",
) -> dict:
    """
    Auto-explain the top N anomalies by absolute deviation magnitude.
    Returns {(sku_id, date): report_text}

    Free-tier safe: adds a 4-second delay between calls (15 req/min limit).
    """
    import time

    top=get_top_anomalies(n, excel_path)
    results = {}
    for _, row in top.iterrows():
        date = str(row["Date"])[:10]
        key  = (row["SKU_ID"], date)
        print(f"  🔄 Explaining {key}...")
        results[key] = explain_anomaly(row["SKU_ID"], date, excel_path)
        time.sleep(4)
    return results



if __name__ == "__main__":
    path = "DATA/sku_demand.csv"
    top1 = get_top_anomalies(1, path)
    sku  = top1.iloc[0]["SKU_ID"]
    date = str(top1.iloc[0]["Date"])[:10]
    print(f"\n📊 Gemini report for {sku} on {date}...\n")
    print(explain_anomaly(sku, date, path))
