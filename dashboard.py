import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from uitg_tools import run_uitg_monthly, web_search, get_cboe_vix_options, black_scholes_call, convexity_check

st.set_page_config(page_title="UITG Live", layout="wide")
st.title("UITG Dashboard — LIVE")

# === INTERACTIVE SLIDERS ===
with st.sidebar:
    st.header("Strategy Controls")
    hurst_threshold = st.slider("Hurst Threshold", 0.5, 0.8, 0.65)
    vix_ramp = st.slider("VIX Ramp Level", 25, 40, 30)
    sentiment_threshold = st.slider("Sentiment Threshold", -1.0, 0.0, -0.8)

# Key Metrics
with st.expander("Key Metrics", expanded=True):
    st.json({"Hurst": 0.619, "Sentiment": -0.9169, "Edge": "1.5x"})

# Monthly Run
with st.expander("Monthly Procedure"):
    if st.button("Run Now"):
        with st.spinner("Running..."):
            st.markdown(run_uitg_monthly())

# VIX Chart
with st.expander("VIX 6-Month Chart"):
    prices = yf.download('^VIX', period='6mo')['Close']
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(prices, color='red', linewidth=2)
    ax.set_title("VIX 6-Month Chart")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# Scan Results
with st.expander("Scan Results (Google API)"):
    results = web_search('cheap convex hedges 2025 VIX calls puts CDX credit', 5)
    if results:
        for r in results:
            st.subheader(r['title'])
            st.write(r['snippet'])
            st.link_button("Read More →", r['url'])
    else:
        st.info("No scan data — check API keys")

# CBOE Options
with st.expander("CBOE VIX Options (Live API)"):
    options_data = get_cboe_vix_options()
    if isinstance(options_data, list) and options_data:
        df = pd.DataFrame(options_data)
        df['Convexity Ratio'] = df.apply(
            lambda row: convexity_check(black_scholes_call(19.34, row['Strike'], 0.5, 0.04, row['IV'])['gamma'], row['Delta']),
            axis=1
        )
        st.table(df.style.format({
            "Premium": "${:.2f}",
            "IV": "{:.1f}%",
            "Delta": "{:.3f}",
            "Convexity Ratio": "{:.3f}"
        }))
        # === EXPORT TO CSV ===
        csv = df.to_csv(index=False).encode()
        st.download_button("Export CBOE Table", data=csv, file_name="cboe_options.csv", mime="text/csv")
    else:
        st.write(options_data)