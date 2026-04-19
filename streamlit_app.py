"""
Streamlit Dashboard for BSTS Trend Engine.
Displays top ETF picks for next-day trading.
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
    .tab-header { font-size: 1.8rem; font-weight: 500; margin: 1rem 0; }
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
    """Render a hero card for the top ETF pick."""
    st.markdown(f"""
    <div class="hero-card">
        <div style="font-size: 1.2rem; opacity: 0.8;">📊 TOP PICK FOR TOMORROW</div>
        <div class="hero-ticker">{ticker}</div>
        <div class="hero-return">{forecast_mean*100:.3f}%</div>
        <div class="hero-range">95% CI: {forecast_lower*100:.3f}% to {forecast_upper*100:.3f}%</div>
    </div>
    """, unsafe_allow_html=True)

def display_forecast_table(universe_data: dict):
    """Display a sortable table of all forecasts in the universe."""
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
        df = pd.DataFrame(rows)
        df = df.sort_values('Forecast', ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No forecast data available.")

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.markdown(f"**Data Source:** `{config.HF_DATA_REPO}`")
st.sidebar.markdown(f"**Results Repo:** `{config.HF_OUTPUT_REPO}`")
st.sidebar.divider()
st.sidebar.markdown("### 📊 BSTS Parameters")
st.sidebar.markdown(f"- Lookback Window: **{config.LOOKBACK_WINDOW} days**")
st.sidebar.markdown(f"- MCMC Samples: **{config.MCMC_SAMPLES}**")
st.sidebar.markdown(f"- MCMC Burn-in: **{config.MCMC_BURN}**")
st.sidebar.divider()

data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")
else:
    st.sidebar.markdown("*No data available*")

# --- Main Content ---
st.markdown('<div class="main-header">📈 P2Quant BSTS Trend Forecaster</div>', unsafe_allow_html=True)
st.markdown('<div style="margin-bottom: 2rem;">Bayesian Structural Time Series with Spike‑and‑Slab Regression</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available. Please run the daily pipeline first.")
    st.stop()

# --- Tabs ---
tab_names = ["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"]
universe_keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]

tabs = st.tabs(tab_names)

for i, (tab, universe_key) in enumerate(zip(tabs, universe_keys)):
    with tab:
        if universe_key in data['universes']:
            universe_data = data['universes'][universe_key]
            top_pick = data.get('top_picks', {}).get(universe_key, {})
            
            if top_pick:
                st.markdown("### 🏆 Top Pick for Tomorrow")
                display_hero_card(
                    top_pick.get('ticker', 'N/A'),
                    top_pick.get('forecast_mean', 0),
                    top_pick.get('forecast_lower', 0),
                    top_pick.get('forecast_upper', 0)
                )
            
            st.markdown("### 📋 All Forecasts")
            display_forecast_table(universe_data)
            
            # Bar chart of forecasts
            st.markdown("### 📊 Forecast Distribution")
            forecasts = {t: d['forecast_mean'] for t, d in universe_data.items() if d.get('forecast_mean') is not None}
            if forecasts:
                sorted_items = sorted(forecasts.items(), key=lambda x: x[1], reverse=True)
                tickers = [item[0] for item in sorted_items]
                values = [item[1] for item in sorted_items]
                colors = ['#667eea' if t == top_pick.get('ticker') else '#a0aec0' for t in tickers]
                fig = go.Figure(go.Bar(x=tickers, y=values, marker_color=colors))
                fig.update_layout(
                    title="Forecasted Next‑Day Returns",
                    xaxis_title="ETF Ticker",
                    yaxis_title="Forecast Return",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No data for {universe_key} universe.")
