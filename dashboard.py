import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from bs4 import BeautifulSoup
from uitg_tools import run_uitg_monthly, web_search, browse_page, convexity_check, black_scholes_call

st.title("UITG Dashboard")

# Key Metrics
with st.expander("Key Metrics", expanded=True):
    st.json({
        "Hurst": 0.619,
        "Sentiment": -0.9169,
        "Edge": "1.5x"
    })

# Monthly Procedure Details
with st.expander("Monthly Procedure"):
    if st.button("Run Monthly Procedure"):
        result = run_uitg_monthly()
        st.markdown(result)

# VIX 6-Month Chart
with st.expander("VIX 6-Month Chart"):
    prices = yf.download('^VIX', period='6mo')['Close']
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(prices)
    st.pyplot(fig)

# Scan Results
with st.expander("Scan Results"):
    results = web_search('cheap convex hedges 2025 VIX calls puts CDX credit', num_results=5)
    for r in results:
        st.write(f"**{r['title']}**")
        st.write(r['snippet'])
        st.link_button("Visit URL", r['url'])

# CBOE Option Suggestions
with st.expander("CBOE Option Suggestions"):
    cboe_url = "https://www.cboe.com/delayed_quotes/vix/quote_table"
    cboe_data = browse_page(cboe_url, "extract OTM VIX calls premiums")
    options = []
    if "Mock" not in cboe_data:
        soup = BeautifulSoup(cboe_data, 'html.parser')
        for row in soup.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 4:
                strike = cells[0].text.strip() if cells[0].text.strip() else ''
                premium = cells[1].text.strip().replace('$', '') if cells[1].text.strip() else ''
                iv = cells[2].text.strip() if cells[2].text.strip() else ''
                delta = cells[3].text.strip() if cells[3].text.strip() else ''
                if strike and premium and delta and float(delta) < 0.5:
                    options.append({"Strike": strike, "Premium": float(premium), "IV": float(iv), "Delta": float(delta)})
    else:
        options = [
            {"Strike": "20", "Premium": 1.11, "IV": 0.25, "Delta": 0.15},
            {"Strike": "22", "Premium": 0.85, "IV": 0.27, "Delta": 0.12}
        ]
    if options:
        df = pd.DataFrame(options)
        df['Convexity Ratio'] = df.apply(
            lambda row: convexity_check(black_scholes_call(19.34, float(row['Strike']), 0.5, 0.04, float(row['IV']))['gamma'], row['Delta']),
            axis=1
        )
        st.table(df.style.format({
            "Premium": "${:.2f}",
            "IV": "{:.1f}%",
            "Delta": "{:.3f}",
            "Convexity Ratio": "{:.3f}"
        }))
    else:
        st.write("No CBOE data available.")