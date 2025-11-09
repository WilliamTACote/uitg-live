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

# Monthly Procedure Results
with st.expander("Monthly Procedure Details"):
    if st.button("Run Monthly Procedure"):
        result = run_uitg_monthly()
        st.markdown(result)

# VIX 6-Month Chart
with st.expander("VIX 6-Month Chart"):
    prices = yf.download('^VIX', period='6mo')['Close']
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(prices, color='red')
    ax.set_title("VIX 6-Month Chart")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# Scan Results
with st.expander("Scan Results"):
    results = web_search('cheap convex hedges 2025 VIX calls puts CDX credit', 5)
    for r in results:
        st.subheader(r['title'])
        st.write(r['snippet'])
        if r['url'].startswith('http'):
            st.link_button("Read More", r['url'])
        else:
            st.caption("_(preview only)_")

# CBOE Option Suggestions
with st.expander("CBOE Option Suggestions"):
    options = []
    cboe_data = browse_page("https://www.cboe.com/delayed_quotes/vix/quote_table", "extract OTM VIX calls premiums")
    if "Mock" not in cboe_data:
        soup = BeautifulSoup(cboe_data, 'html.parser')
        for row in soup.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 4:
                strike = cells[0].get_text(strip=True)
                premium = cells[1].get_text(strip=True).replace('$', '')
                iv = cells[2].get_text(strip=True)
                delta = cells[3].get_text(strip=True)
                if strike and premium and delta:
                    try:
                        if float(delta) < 0.5:
                            options.append({
                                "Strike": strike,
                                "Premium": float(premium),
                                "IV": float(iv),
                                "Delta": float(delta)
                            })
                    except:
                        continue
    else:
        options = [
            {"Strike": "20", "Premium": 1.11, "IV": 0.25, "Delta": 0.15},
            {"Strike": "22", "Premium": 0.85, "IV": 0.27, "Delta": 0.12}
        ]

    if options:
        df = pd.DataFrame(options)
        df['Convexity Ratio'] = df.apply(
            lambda r: convexity_check(
                black_scholes_call(19.34, float(r['Strike']), 0.5, 0.04, float(r['IV']))['gamma'],
                r['Delta']
            ), axis=1
        )
        st.table(df)
    else:
        st.write("No CBOE data available.")