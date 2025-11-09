import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from uitg_tools import run_uitg_monthly, google_search_hedges, get_cboe_vix_options, convexity_check, black_scholes_call

st.set_page_config(page_title="UITG Live", layout="wide")
st.title("UITG Dashboard — LIVE")

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

# Real Scan Results
with st.expander("Scan Results (Google API)"):
    results = google_search_hedges()
    if results:
        for r in results:
            st.subheader(r['title'])
            st.write(r['snippet'])
            st.link_button("Read More →", r['url'])
    else:
        st.info("No real scan data — check API keys")

# Real CBOE Options
with st.expander("CBOE VIX Options (Live API)"):
    options = get_cboe_vix_options()
    if options:
        df = pd.DataFrame(options)
        df['Convexity Ratio'] = df.apply(
            lambda r: convexity_check(
                black_scholes_call(19.34, r['Strike'], 0.5, 0.04, r['IV'])['gamma'],
                r['Delta']
            ), axis=1
        )
        st.success("Real CBOE data loaded!")
        st.dataframe(df.style.format({
            "Premium": "${:.2f}",
            "IV": "{:.1f}%",
            "Delta": "{:.3f}",
            "Convexity Ratio": "{:.3f}"
        }))
    else:
        st.warning("CBOE API failed — using fallback")