"""
Streamlit Dashboard for BSTS Trend Engine.
Displays Daily Active Trading and Shrinking Windows forecasts.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from huggingface_hub import HfApi, hf_hub_download
import json
import config

st.set_page_config(
    page_title="P2Quant BSTS Trend Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; margin-bottom: 0.5rem; }
    .hero-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                 border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .hero-ticker { font-size: 4rem; font-weight: 800; }
    .hero-return { font-size: 2.5rem; font-weight: 600; }
    .hero-range { font-size: 1.2rem; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    """Fetch the most recent result file from HF dataset."""
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.endswith('.json')], reverse=True)
        if not json_files:
            return None
        local_path = hf_hub_download(
            repo_id=config.HF_OUTPUT_REPO,
            filename=json_files[0],
            repo_type="dataset",
            token=config.HF_TOKEN,
            cache_dir="./hf_cache"
        )
        with open(local_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def display_hero_card(ticker: str, forecast_mean: float, forecast_lower: float, forecast_upper: float):
    st.markdown(f"""
    <div class="hero-card">
        <div style="font-size: 1.2rem; opacity: 0.8;">📊 TOP PICK FOR TOMORROW</div>
        <div class="hero-ticker">{ticker}</div>
        <div class="hero-return">{forecast_mean*100:.3f}%</div>
        <div class="hero-range">95% CI: {forecast_lower*100:.3f}% to {forecast_upper*100:.3f}%</div>
    </div>
    """, unsafe_allow_html=True)

def display_forecast_table(universe_data: dict):
    rows = []
    for ticker, data in universe_data.items():
        if data.get('forecast_mean') is not None:
            rows.append({
                'Ticker': ticker,
                'Forecast': f"{data['forecast_mean']*100:.3f}%",
                'Lower 95%': f"{data['forecast_lower']*100:.3f}%",
                'Upper 95%': f"{data['forecast_upper']*100:.3f}%"
            })
    if rows:
        df = pd.DataFrame(rows).sort_values('Forecast', ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No forecast data available.")

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.markdown(f"**Data Source:** `{config.HF_DATA_REPO}`")
st.sidebar.markdown(f"**Results Repo:** `{config.HF_OUTPUT_REPO}`")
st.sidebar.divider()
st.sidebar.markdown("### 📊 BSTS Parameters")
st.sidebar.markdown(f"- Daily Lookback: **{config.LOOKBACK_WINDOW} days**")
st.sidebar.markdown(f"- Shrinking Windows: **{', '.join(map(str, config.SHRINKING_WINDOW_START_YEARS))}**")
st.sidebar.divider()

data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")
else:
    st.sidebar.markdown("*No data available*")

# --- Main Content ---
st.markdown('<div class="main-header">📈 P2Quant BSTS Trend Forecaster</div>', unsafe_allow_html=True)
st.markdown('<div style="margin-bottom: 2rem;">Bayesian Structural Time Series with Macro Predictors</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available. Please run the daily pipeline first.")
    st.stop()

# --- Top-Level Tabs ---
main_tab1, main_tab2 = st.tabs(["📋 Daily Active Trading", "📆 Shrinking Windows"])

# ------------------------------
# DAILY ACTIVE TRADING TAB
# ------------------------------
with main_tab1:
    daily_data = data['daily_active']
    top_picks = daily_data['top_picks']
    universes_data = daily_data['universes']
    
    subtab_names = ["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"]
    subtab_keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]
    subtabs = st.tabs(subtab_names)
    
    for subtab, key in zip(subtabs, subtab_keys):
        with subtab:
            if key in universes_data:
                universe_dict = universes_data[key]
                pick = top_picks.get(key, {})
                if pick:
                    st.markdown("### 🏆 Top Pick for Tomorrow")
                    display_hero_card(
                        pick.get('ticker', 'N/A'),
                        pick.get('forecast_mean', 0),
                        pick.get('forecast_lower', 0),
                        pick.get('forecast_upper', 0)
                    )
                st.markdown("### 📋 All Forecasts")
                display_forecast_table(universe_dict)
                
                # Bar chart
                forecasts = {t: d['forecast_mean'] for t, d in universe_dict.items() if d.get('forecast_mean') is not None}
                if forecasts:
                    sorted_items = sorted(forecasts.items(), key=lambda x: x[1], reverse=True)
                    tickers = [item[0] for item in sorted_items]
                    values = [item[1] for item in sorted_items]
                    colors = ['#667eea' if t == pick.get('ticker') else '#a0aec0' for t in tickers]
                    fig = go.Figure(go.Bar(x=tickers, y=values, marker_color=colors))
                    fig.update_layout(
                        title="Forecasted Next‑Day Returns",
                        xaxis_title="ETF Ticker",
                        yaxis_title="Forecast Return",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No data for {key} universe.")

# ------------------------------
# SHRINKING WINDOWS TAB
# ------------------------------
with main_tab2:
    shrinking = data.get('shrinking_windows', {})
    if not shrinking:
        st.warning("No shrinking windows data available.")
        st.stop()
    
    # Subtabs per universe
    st.markdown("### Forecasts from Different Historical Start Dates")
    subtab_names_sw = ["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"]
    subtab_keys_sw = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]
    sw_tabs = st.tabs(subtab_names_sw)
    
    for subtab, universe_key in zip(sw_tabs, subtab_keys_sw):
        with subtab:
            rows = []
            windows_sorted = sorted(shrinking.keys(), reverse=True)  # newest first
            for window_label in windows_sorted:
                win_data = shrinking[window_label]
                top = win_data['top_picks'].get(universe_key, {})
                if top:
                    rows.append({
                        'Window': window_label,
                        'Top Pick': top.get('ticker', 'N/A'),
                        'Forecast': f"{top.get('forecast_mean', 0)*100:.3f}%",
                        '95% CI Lower': f"{top.get('forecast_lower', 0)*100:.3f}%",
                        '95% CI Upper': f"{top.get('forecast_upper', 0)*100:.3f}%"
                    })
            if rows:
                df_sw = pd.DataFrame(rows)
                st.dataframe(df_sw, use_container_width=True, hide_index=True)
                
                # Optional: line chart showing forecast trend across windows
                df_chart = df_sw.copy()
                df_chart['Forecast_val'] = df_chart['Forecast'].str.rstrip('%').astype(float)
                fig = go.Figure(go.Scatter(
                    x=df_chart['Window'], y=df_chart['Forecast_val'],
                    mode='lines+markers', text=df_chart['Top Pick'],
                    line=dict(color='#667eea', width=3)
                ))
                fig.update_layout(
                    title=f"{universe_key} – Top Pick Forecast by Window",
                    xaxis_title="Window Start Year",
                    yaxis_title="Forecast Return (%)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No shrinking window data for {universe_key}.")
