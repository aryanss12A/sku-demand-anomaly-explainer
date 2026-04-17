import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings("ignore")


def run_prophet_detection(df_sku: pd.DataFrame, sensitivity: float = 0.05):
    """
    Fit Prophet model and return rows where actual demand
    falls outside the uncertainty interval.
    """
    df_prophet = df_sku[["date", "demand"]].rename(
        columns={"date": "ds", "demand": "y"}
    )
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

    model = Prophet(
        interval_width=1 - sensitivity,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model.add_country_holidays(country_name="IN")
    model.fit(df_prophet)

    forecast = model.predict(df_prophet[["ds"]])

    result = df_prophet.merge(
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds"
    )

    result["is_anomaly_prophet"] = (
        (result["y"] > result["yhat_upper"]) |
        (result["y"] < result["yhat_lower"])
    )
    result["anomaly_type"] = np.where(
        result["y"] > result["yhat_upper"], "spike",
        np.where(result["y"] < result["yhat_lower"], "dip", "normal")
    )
    result["deviation_pct"] = (
        (result["y"] - result["yhat"]) / result["yhat"] * 100
    ).round(2)

    return result, model, forecast


def run_isolation_forest(df_sku: pd.DataFrame, contamination: float = 0.05):
    """
    Use Isolation Forest as a second-pass validator on feature-engineered data.
    """
    features = df_sku[["demand", "price", "is_promotion",
                        "stock_available", "competitor_oos"]].copy()
    features["day_of_week"] = pd.to_datetime(df_sku["date"]).dt.dayofweek
    features["month"] = pd.to_datetime(df_sku["date"]).dt.month

    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    preds = iso.fit_predict(features)
    scores = iso.score_samples(features)

    return (preds == -1), scores


def detect_anomalies(df_sku: pd.DataFrame, sensitivity: float = 0.05):
    """
    Master function: combines Prophet + IsoForest with AND/OR logic.
    Returns enriched DataFrame with anomaly flags and metadata.
    """
    prophet_result, model, forecast = run_prophet_detection(df_sku, sensitivity)
    iso_flags, iso_scores = run_isolation_forest(df_sku, contamination=sensitivity)

    df_sku = df_sku.copy().reset_index(drop=True)
    prophet_result = prophet_result.reset_index(drop=True)

    df_sku["yhat"] = prophet_result["yhat"]
    df_sku["yhat_lower"] = prophet_result["yhat_lower"]
    df_sku["yhat_upper"] = prophet_result["yhat_upper"]
    df_sku["prophet_anomaly"] = prophet_result["is_anomaly_prophet"]
    df_sku["anomaly_type"] = prophet_result["anomaly_type"]
    df_sku["deviation_pct"] = prophet_result["deviation_pct"]
    df_sku["iso_anomaly"] = iso_flags
    df_sku["iso_score"] = iso_scores

    # Final anomaly: flagged by EITHER model (union for recall)
    df_sku["is_anomaly"] = df_sku["prophet_anomaly"] | df_sku["iso_anomaly"]

    # Confidence: flagged by BOTH models
    df_sku["high_confidence"] = df_sku["prophet_anomaly"] & df_sku["iso_anomaly"]

    anomalies = df_sku[df_sku["is_anomaly"]].copy()
    return df_sku, anomalies, model, forecast


def load_raw_data(path):
    df = pd.read_csv(path)
    df = df.rename(columns={
        'sku_id': 'SKU_ID',
        'date': 'Date',
        'demand': 'Demand',
        'price': 'Price',
        'is_promotion': 'Is_Promotion',
        'stock_available': 'Stock_Available',
        'competitor_oos': 'Competitor_OOS',
        'category': 'Category'
    })
    df['Date'] = pd.to_datetime(df['Date'])
    # For detection, need lowercase columns
    df_for_detect = df.rename(columns={
        'Date': 'date', 'Demand': 'demand', 'Price': 'price',
        'Is_Promotion': 'is_promotion', 'Stock_Available': 'stock_available',
        'Competitor_OOS': 'competitor_oos'
    })
    # Add forecast columns by running detection for each SKU
    forecast_cols = []
    for sku in df['SKU_ID'].unique():
        sku_df = df_for_detect[df_for_detect['SKU_ID'] == sku].copy()
        df_full, anomalies, model, forecast = detect_anomalies(sku_df, sensitivity=0.05)
        df_full = df_full.rename(columns={'yhat': 'Forecast', 'yhat_lower': 'Forecast_Lower', 'yhat_upper': 'Forecast_Upper'})
        df_full['SKU_ID'] = sku  # ensure SKU_ID
        forecast_cols.append(df_full[['SKU_ID', 'date', 'Forecast', 'Forecast_Lower', 'Forecast_Upper', 'is_anomaly', 'anomaly_type', 'deviation_pct', 'high_confidence', 'iso_score']].rename(columns={'date': 'Date'}))
    forecast_df = pd.concat(forecast_cols)
    df = df.merge(forecast_df, on=['SKU_ID', 'Date'], how='left')
    return df


def load_anomaly_summary(path):
    raw = load_raw_data(path)
    anomalies = raw[raw['is_anomaly'] == True].copy()
    anomalies = anomalies.rename(columns={'anomaly_type': 'Anomaly_Type', 'deviation_pct': 'Deviation_Pct', 'high_confidence': 'High_Confidence', 'iso_score': 'ISO_Score'})
    # Capitalize anomaly types for consistency
    anomalies['Anomaly_Type'] = anomalies['Anomaly_Type'].str.capitalize()
    anomalies['Year'] = anomalies['Date'].dt.year
    anomalies['Event_Name'] = None
    anomalies['Region'] = 'IN'
    return anomalies


def load_monthly_kpis(path):
    raw = load_raw_data(path)
    monthly = raw.groupby([raw['Date'].dt.to_period('M'), 'SKU_ID']).agg(
        Total_Demand=('Demand', 'sum'),
        Anomaly_Count=('is_anomaly', 'sum')
    ).reset_index()
    monthly['Month'] = monthly['Date'].dt.strftime('%Y-%m')
    monthly = monthly.drop('Date', axis=1)
    return monthly


def load_event_impact(path):
    # Dummy data since no events in CSV
    return pd.DataFrame({
        'Event_Name': ['Festival', 'Promotion'],
        'Anomaly_Type': ['Spike', 'Dip'],
        'Avg_Deviation': [10.0, -5.0]
    })


def compute_global_kpis(path):
    raw = load_raw_data(path)
    anom = load_anomaly_summary(path)
    skus = raw['SKU_ID'].unique().tolist()
    total_records = len(raw)
    total_demand = raw['Demand'].sum()
    total_revenue = (raw['Demand'] * raw['Price']).sum() / 1e6
    total_anomalies = len(anom)
    anomaly_rate_pct = round(total_anomalies / total_records * 100, 1)
    spikes = len(anom[anom['Anomaly_Type'] == 'spike'])
    dips = len(anom[anom['Anomaly_Type'] == 'dip'])
    date_range = f"{raw['Date'].min().strftime('%Y-%m-%d')} to {raw['Date'].max().strftime('%Y-%m-%d')}"
    return {
        'skus': skus,
        'total_records': total_records,
        'total_demand': total_demand,
        'total_revenue_m': round(total_revenue, 1),
        'total_anomalies': total_anomalies,
        'anomaly_rate_pct': anomaly_rate_pct,
        'spikes': spikes,
        'dips': dips,
        'date_range': date_range
    }


def get_surrounding_context(sku_id, date_str, path):
    raw = load_raw_data(path)
    sku_df = raw[raw['SKU_ID'] == sku_id]
    date = pd.to_datetime(date_str)
    context = sku_df[(sku_df['Date'] >= date - pd.Timedelta(days=7)) & (sku_df['Date'] <= date + pd.Timedelta(days=3))]
    return context[['Date', 'Demand', 'Price', 'Is_Promotion']]


def get_top_anomalies(n, path):
    anom = load_anomaly_summary(path)
    top = anom.reindex(anom['Deviation_Pct'].abs().nlargest(n).index)
    return top

