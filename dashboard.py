import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from bs4 import BeautifulSoup
from uitg_tools import run_uitg_monthly, web_search, browse_page, convexity_check, black_scholes_call

st.set_page_config(page_title="UITG Dashboard", layout="wide")
st.title("UITG Dashboard")

with st.expander("Key Metrics", expanded=True):
    st.json({"Hurst": 0.619, "Sentiment": -0.9169, "Edge": "1.5x"})

with st.expander("Monthly Procedure"):
    if st.button("Run Monthly Procedure"):
        with st.spinner("Running..."):
            st.markdown(run_uitg_monthly())

with st.expander("VIX 6-Month Chart"):
    prices = yf.download('^VIX', period='6mo')['Close']
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(prices, color='red')
    ax.set_title("VIX 6-Month Chart")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with st.expander("Scan Results"):
    results = web_search('cheap convex hedges 2025 VIX calls puts CDX credit', 5)
    for r in results:
        st.subheader(r['title'])
        st.write(r['snippet'])
        st.link_button("Visit URL", r['url'])

with st.expander("CBOE Option Suggestions"):
    data = browse_page("https://www.cboe.com/delayed_quotes/vix/quote_table", "extract OTM VIX calls premiums")
    if "Mock" in data:
        st.info("Using fallback data")
        options = [{"Strike": "20", "Premium": 1.11, "IV": 0.25, "Delta": 0.15},
                   {"Strike": "22", "Premium": 0.85, "IV": 0.27, "Delta": 0.12}]
    else:
        st.success("Real CBOE data!")
        st.write(data)
        options = []

    if options:
        df = pd.DataFrame(options)
        df['Convexity Ratio'] = df.apply(
            lambda r: convexity_check(
                black_scholes_call(19.34, float(r['Strike']), 0.5, 0.04, float(r['IV']))['gamma'],
                r['Delta']
            ), axis=1
        )
        st.table(df.style.format({"Premium": "${:.2f}", "IV": "{:.2f}", "Delta": "{:.3f}", "Convexity Ratio": "{:.3f}"}))