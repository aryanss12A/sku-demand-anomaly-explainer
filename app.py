"""
app.py — Streamlit Dashboard
Reads ALL data directly from SKU_Anomaly_Dashboard_PowerBI.xlsx.
AI reports powered by Google Gemini (free tier).
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import google.genai as genai
import os
from dotenv import load_dotenv

from SRC.anomaly_detector import (
    load_raw_data, load_anomaly_summary,
    load_monthly_kpis, load_event_impact,
    compute_global_kpis, get_surrounding_context,
    get_top_anomalies,
)
from SRC.explainer import explain_anomaly

load_dotenv()
EXCEL_PATH = "DATA/sku_demand.csv"

# ── Gemini API key check ──────────────────────────────────────
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_KEY:
    pass  # No configure needed for google.genai

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="SKU Anomaly Explainer",
    page_icon="📦", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main { background-color: #0f1117; }
  .report-box {
    background: #1a1f2e; border-left: 3px solid #4c8bf5;
    border-radius: 8px; padding: 1.5rem;
    font-size: 0.92rem; line-height: 1.7;
  }
</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────
@st.cache_data
def load_all():
    raw     = load_raw_data(EXCEL_PATH)
    anomaly = load_anomaly_summary(EXCEL_PATH)
    monthly = load_monthly_kpis(EXCEL_PATH)
    events  = load_event_impact(EXCEL_PATH)
    kpis    = compute_global_kpis(EXCEL_PATH)
    return raw, anomaly, monthly, events, kpis

raw_df, anom_df, monthly_df, event_df, kpis = load_all()

COLORS = {
    "SKU_001": "#3B82F6", "SKU_002": "#10B981",
    "SKU_003": "#F97316", "SKU_004": "#8B5CF6",
}

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.title("SKU Anomaly\nExplainer")
    st.markdown(f"**Data:** `{EXCEL_PATH}`")
    st.markdown(f"📅 {kpis['date_range']}")
    st.markdown(f"📊 {kpis['total_records']:,} records")
    st.markdown("---")

    selected_sku = st.selectbox(
        "🔍 Select SKU",
        options=kpis["skus"],
        format_func=lambda s: f"{s} · {raw_df[raw_df['SKU_ID']==s]['Category'].iloc[0]}",
    )
    anom_type_filter = st.selectbox("Anomaly Type", ["All", "Spike", "Dip"])
    year_filter      = st.selectbox("Year", ["All", "2022", "2023", "2024"])

    st.markdown("---")
    st.caption("📦 Prophet + Isolation Forest + Gemini AI")
    st.caption(f"Built by Aryan | {kpis['total_anomalies']} anomalies detected")

# ── Filters ───────────────────────────────────────────────────
def apply_filters(df):
    if anom_type_filter != "All":
        df = df[df["Anomaly_Type"] == anom_type_filter]
    if year_filter != "All":
        df = df[df["Year"] == int(year_filter)]
    return df

filtered_anom = apply_filters(anom_df)

# ── Title ─────────────────────────────────────────────────────
st.title("📦 SKU-Level Demand Anomaly Explainer")
st.markdown(
    f"Data from `{EXCEL_PATH}` · **{kpis['total_anomalies']} anomalies** · "
    f"**{kpis['date_range']}** · Powered by **Google Gemini**"
)
st.markdown("---")

# ── KPI Strip ─────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Demand",    f"{kpis['total_demand']:,} units")
c2.metric("Est. Revenue",    f"₹{kpis['total_revenue_m']}M")
c3.metric("Anomalies",       kpis["total_anomalies"],
          delta=f"{kpis['anomaly_rate_pct']}% of days")
c4.metric("🔺 Spikes",       kpis["spikes"])
c5.metric("🔻 Dips",         kpis["dips"])
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Demand Chart", "🚨 Anomaly Table",
    "📅 Monthly KPIs", "🎯 Event Impact", "🤖 AI Explainer",
])

# ── TAB 1: Demand Chart ───────────────────────────────────────
with tab1:
    st.subheader(f"📈 Daily Demand vs Forecast — {selected_sku}")
    sku_raw  = raw_df[raw_df["SKU_ID"] == selected_sku].copy()
    sku_anom = anom_df[anom_df["SKU_ID"] == selected_sku].copy()
    spikes   = sku_anom[sku_anom["Anomaly_Type"] == "Spike"]
    dips     = sku_anom[sku_anom["Anomaly_Type"] == "Dip"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sku_raw["Date"], y=sku_raw["Forecast_Upper"],
        mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=sku_raw["Date"], y=sku_raw["Forecast_Lower"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(76,139,245,.12)",
        name="Forecast Band",
    ))
    fig.add_trace(go.Scatter(
        x=sku_raw["Date"], y=sku_raw["Forecast"],
        mode="lines",
        line=dict(color=COLORS[selected_sku], width=1.5, dash="dot"),
        name="Prophet Forecast",
    ))
    fig.add_trace(go.Scatter(
        x=sku_raw["Date"], y=sku_raw["Demand"],
        mode="lines", line=dict(color="#a0aec0", width=1.5),
        name="Actual Demand",
    ))
    fig.add_trace(go.Scatter(
        x=spikes["Date"], y=spikes["Demand"], mode="markers",
        marker=dict(color="#EF4444", size=9, symbol="triangle-up",
                    line=dict(color="white", width=1)),
        name="🔺 Spike",
    ))
    fig.add_trace(go.Scatter(
        x=dips["Date"], y=dips["Demand"], mode="markers",
        marker=dict(color="#F97316", size=9, symbol="triangle-down",
                    line=dict(color="white", width=1)),
        name="🔻 Dip",
    ))
    fig.update_layout(
        template="plotly_dark", height=450,
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#1e2130", title="Units"),
        margin=dict(t=20, l=0, r=0, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Monthly Demand by SKU")
    fig2 = px.bar(
        monthly_df, x="Month", y="Total_Demand", color="SKU_ID",
        color_discrete_map=COLORS, barmode="stack",
        template="plotly_dark", height=300,
    )
    fig2.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        margin=dict(t=10, l=0, r=0, b=0),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── TAB 2: Anomaly Table ──────────────────────────────────────
with tab2:
    st.subheader(f"🚨 Anomaly Summary — {len(filtered_anom)} records")
    st.caption("Source: 🚨 Anomaly Summary sheet in your Excel workbook")

    col_a, col_b = st.columns(2)
    with col_a:
        counts = anom_df.groupby("SKU_ID")["SKU_ID"].count().reset_index(name="Count")
        fig_bar = px.bar(counts, x="SKU_ID", y="Count", color="SKU_ID",
                         color_discrete_map=COLORS, template="plotly_dark",
                         height=220, title="Anomalies per SKU")
        fig_bar.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                               margin=dict(t=30, l=0, r=0, b=0), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_b:
        fig_pie = px.pie(
            anom_df, names="Anomaly_Type",
            color="Anomaly_Type",
            color_discrete_map={"Spike":"#EF4444","Dip":"#F97316"},
            template="plotly_dark", height=220, title="Spike vs Dip Split",
        )
        fig_pie.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                               margin=dict(t=30, l=0, r=0, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    display = filtered_anom[[
        "Date","SKU_ID","Category","Region","Anomaly_Type",
        "Demand","Forecast","Deviation_Pct","High_Confidence","Event_Name","ISO_Score",
    ]].sort_values("Deviation_Pct", key=abs, ascending=False).copy()
    display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(
        display.style.background_gradient(subset=["Deviation_Pct"], cmap="RdYlGn"),
        use_container_width=True, height=400,
    )

# ── TAB 3: Monthly KPIs ───────────────────────────────────────
with tab3:
    st.subheader("📅 Monthly KPI Tracker")
    st.caption("Source: 📅 Monthly KPIs sheet in your Excel workbook")
    sku_sel = st.multiselect("Filter SKUs", kpis["skus"], default=kpis["skus"])
    m_filt  = monthly_df[monthly_df["SKU_ID"].isin(sku_sel)]

    col1, col2 = st.columns(2)
    with col1:
        fig_m1 = px.line(m_filt, x="Month", y="Total_Demand", color="SKU_ID",
                          color_discrete_map=COLORS, template="plotly_dark",
                          height=280, title="Monthly Total Demand")
        fig_m1.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                               margin=dict(t=30, l=0, r=0, b=0))
        st.plotly_chart(fig_m1, use_container_width=True)
    with col2:
        fig_m2 = px.bar(m_filt, x="Month", y="Anomaly_Count", color="SKU_ID",
                         color_discrete_map=COLORS, barmode="stack",
                         template="plotly_dark", height=280, title="Monthly Anomaly Count")
        fig_m2.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                               margin=dict(t=30, l=0, r=0, b=0))
        st.plotly_chart(fig_m2, use_container_width=True)

    st.dataframe(
        m_filt.style.background_gradient(
            subset=["Anomaly_Count","Total_Demand"], cmap="Blues"
        ),
        use_container_width=True, height=350,
    )

# ── TAB 4: Event Impact ───────────────────────────────────────
with tab4:
    st.subheader("🎯 Event Impact Analysis")
    st.caption("Source: 🎯 Event Impact sheet in your Excel workbook")

    fig_ev = px.bar(
        event_df.sort_values("Avg_Deviation", key=abs, ascending=True),
        x="Avg_Deviation", y="Event_Name", color="Anomaly_Type",
        orientation="h", template="plotly_dark", height=420,
        color_discrete_map={"Spike":"#EF4444","Dip":"#F97316"},
        title="Average Demand Deviation % by Event",
        text="Avg_Deviation",
    )
    fig_ev.update_traces(texttemplate="%{text:+.1f}%", textposition="outside")
    fig_ev.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        margin=dict(t=30, l=200, r=60, b=0),
    )
    st.plotly_chart(fig_ev, use_container_width=True)
    st.dataframe(
        event_df.sort_values("Avg_Deviation", key=abs, ascending=False),
        use_container_width=True, height=300,
    )

# ── TAB 5: AI Explainer (Gemini) ─────────────────────────────
with tab5:
    st.subheader("🤖 Gemini AI Root Cause Explainer")
    st.markdown(
        "Pulls real anomaly context from your Excel, sends to **Google Gemini** "
        "(free tier — no billing required)."
    )

    # API key status
    if GEMINI_KEY:
        st.success("✅ GEMINI_API_KEY detected — live reports enabled.")
    else:
        st.warning(
            "⚠️ **GEMINI_API_KEY not set.**\n\n"
            "1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)\n"
            "2. Click **Create API Key** (free, no credit card)\n"
            "3. Add `GEMINI_API_KEY=your_key` to your `.env` file\n"
            "4. Restart the app"
        )

    top_anom = get_top_anomalies(20, EXCEL_PATH)
    options  = top_anom.apply(
        lambda r: f"{str(r['Date'])[:10]} | {r['SKU_ID']} | "
                  f"{r['Anomaly_Type']} | {r['Deviation_Pct']:+.1f}%",
        axis=1,
    ).tolist()

    selected_label = st.selectbox("Select Anomaly to Explain", options)
    idx            = options.index(selected_label)
    sel_row        = top_anom.iloc[idx]

    col_info, col_report = st.columns([1, 1.5])

    with col_info:
        st.markdown("**Selected Anomaly Details**")
        st.markdown(f"""
| Field | Value |
|---|---|
| SKU ID | `{sel_row['SKU_ID']}` |
| Category | {sel_row['Category']} |
| Date | {str(sel_row['Date'])[:10]} |
| Type | **{sel_row['Anomaly_Type']}** |
| Actual Demand | {sel_row['Demand']:,} |
| Forecast | {sel_row['Forecast']:,} |
| Deviation | **{sel_row['Deviation_Pct']:+.2f}%** |
| ISO Score | {sel_row['ISO_Score']} |
| High Confidence | {sel_row['High_Confidence']} |
| Event | {sel_row['Event_Name'] or '—'} |
""")
        date_str = str(sel_row["Date"])[:10]
        context  = get_surrounding_context(sel_row["SKU_ID"], date_str, path=EXCEL_PATH)
        context["Date"] = context["Date"].dt.strftime("%Y-%m-%d")
        st.caption("7-Day Surrounding Context")
        st.dataframe(context, use_container_width=True, height=220)

    with col_report:
        if st.button("🤖 Generate Root Cause Report", type="primary"):
            if not GEMINI_KEY:
                st.error("GEMINI_API_KEY not set — see instructions above.")
            else:
                with st.spinner("Gemini is analyzing the anomaly…"):
                    report = explain_anomaly(
                        sel_row["SKU_ID"],
                        str(sel_row["Date"])[:10],
                        EXCEL_PATH,
                    )
                st.markdown("### 📄 Root Cause Report")
                st.markdown(
                    f'<div class="report-box">{report}</div>',
                    unsafe_allow_html=True,
                )
                st.download_button(
                    "⬇️ Download Report",
                    data=report,
                    file_name=f"gemini_report_{sel_row['SKU_ID']}_{str(sel_row['Date'])[:10]}.txt",
                    mime="text/plain",
                )