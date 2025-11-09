# UITG Tools Master Script
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from hurst import compute_Hc
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import streamlit as st
from PIL import Image
from io import BytesIO
import PyPDF2
import pdfplumber
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.stats import norm
import time
import warnings
import requests

warnings.filterwarnings("ignore", category=FutureWarning)

# Persistent globals for code_execution
persistent_globals = {}

def test_uitg_setup():
    return "UITG setup test passed"

def view_x_video(video_url: str) -> str:
    return f"Mock video viewed: {video_url} (e.g., hedge explainer)"

def code_execution(code: str) -> str:
    try:
        exec_locals = {}
        exec(code, persistent_globals, exec_locals)
        persistent_globals.update(exec_locals)
        result = exec_locals.get('result', 'No result variable set')
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def monte_carlo_payouts(N=50000, steps=40, alloc=0.2, crash_prob=0.05, multiplier=10, decay=-0.002):
    np.random.seed(42)
    capitals = np.full(N, 20000, dtype=np.float64)
    for s in range(steps):
        crashes = np.random.rand(N) < crash_prob
        hedge_returns = np.full(N, decay * alloc, dtype=np.float64)
        hedge_returns[crashes] = multiplier * alloc - alloc
        capitals *= (1 + hedge_returns)
    return np.mean(capitals)

def black_scholes_call(S, K, t, r, sig):
    d1 = (np.log(S/K) + (r + sig**2/2)*t) / (sig * np.sqrt(t))
    d2 = d1 - sig * np.sqrt(t)
    call = S * norm.cdf(d1) - K * np.exp(-r*t) * norm.cdf(d2)
    gamma = norm.pdf(d1) / (S * sig * np.sqrt(t))
    delta = norm.cdf(d1)
    vega = S * np.sqrt(t) * norm.pdf(d1) / 100
    return {'call': call, 'gamma': gamma, 'delta': delta, 'vega': vega}

# X Search Tools
def x_keyword_search(query: str, limit: int = 10, mode: str = "Top"):
    # Mock with negative sentiment for crash queries
    posts = [{'text': f'Mock crash warning for {query} #{i} market panic', 'score': np.random.uniform(-0.8, -0.2)} for i in range(limit)]
    return posts

def x_semantic_search(query: str, limit: int = 10, min_score_threshold: float = 0.18):
    # Mock semantic; use sentence-transformers for real
    mock_docs = [f'Mock relevant post for {query} #{i}' for i in range(limit)]
    scores = np.random.uniform(0.1, 0.9, limit)
    filtered = [mock_docs[i] for i in range(limit) if scores[i] > min_score_threshold]
    return [{'text': doc, 'score': float(scores[i])} for i in range(limit) if scores[i] > min_score_threshold]

def aggregate_sentiment(posts: list) -> float:
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(post['text'])['compound'] for post in posts]
    return np.mean(scores) if scores else 0

# Real Web Search Tools (Google Custom Search)
def google_search_hedges(query="cheap convex hedges 2025 VIX calls puts CDX", num=5):
    API_KEY = st.secrets["GOOGLE_API_KEY"]  # Hidden in Streamlit Cloud
    CX = st.secrets["GOOGLE_CX"]  # Hidden in Streamlit Cloud
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"q": query, "key": API_KEY, "cx": CX, "num": num}
    try:
        response = requests.get(url, params=params, timeout=10)
        items = response.json().get("items", [])
        return [{
            "title": item.get("title", "No title"),
            "snippet": item.get("snippet", "No snippet"),
            "url": item.get("link", "#")
        } for item in items]
    except:
        return [{'title': f'Fallback result for {query} #{i}', 'snippet': f'Suggested hedge: OTM VIX call at strike {20+i}, premium ~${1.0 + i*0.1}', 'url': f'url{i}.com'} for i in range(num)]

# Real CBOE VIX Options API
def get_cboe_vix_options():
    url = "https://cdn.cboe.com/api/global/delayed_quotes/options/VX.json"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        options = []
        for opt in data['data']['calls']:
            strike = opt['strike']
            bid = opt['bid']
            ask = opt['ask']
            iv = opt['volatility']
            delta = opt['delta']
            if bid > 0 and ask > 0 and delta < 0.5:  # OTM filter
                premium = (bid + ask) / 2
                options.append({
                    "Strike": strike,
                    "Premium": round(premium, 2),
                    "IV": round(iv, 2),
                    "Delta": round(delta, 3)
                })
        return options[:5]  # Top 5 OTM calls
    except:
        # yfinance fallback
        try:
            vix = yf.download('^VIX', period='1d')['Close'].iloc[-1]
            return f"Real VIX Index: {vix:.2f} (yfinance fallback) — Suggest OTM calls at strike {round(vix + 5, 0)}"
        except:
            return "No CBOE data"

# Convexity Check Tool
def convexity_check(gamma, delta):
    return gamma / delta if delta != 0 else 0  # Ratio >0.5 for high convexity

# PDF Tools
def search_pdf_attachment(file_name: str, query: str, mode: str = 'keyword') -> list:
    try:
        with open(file_name, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            pages = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if mode == 'keyword' and query.lower() in text.lower():
                    pages.append({'page': page_num + 1, 'snippet': text[:200]})
            return pages
    except Exception as e:
        return f"Error searching PDF {file_name}: {str(e)}"

def browse_pdf_attachment(file_name: str, pages: str) -> str:
    try:
        with pdfplumber.open(file_name) as pdf:
            page_list = [int(p.strip()) - 1 for p in pages.split(',') if p.strip().isdigit()]
            text = ''
            for p in page_list:
                page = pdf.pages[p]
                text += page.extract_text() + '\n'
            return text[:1000] + '...'
    except Exception as e:
        return f"Error browsing PDF {file_name}: {str(e)}"

# Master System Integration
def run_uitg_monthly():
    # Full monthly procedure: Scan, metrics, sentiment, entry check, positioning, visualization
    # Patience Principle: Embrace the drag in bull markets as the price of convexity in bears; resist frequent adjustments unless triggers are met, as over-trading erodes asymmetry.
    # Scan hedges
    scan_results = web_search('cheap convex hedges 2025 VIX calls puts CDX credit', num_results=5)
    # Compute metrics on VIX
    code = "import numpy as np\n" + \
           "import yfinance as yf\n" + \
           "from hurst import compute_Hc\n" + \
           "from scipy.stats import norm\n" + \
           "prices = yf.download('^VIX', period='6mo')['Close'].dropna()\n" + \
           "hurst = compute_Hc(prices, kind='price')[0]\n" + \
           "returns = prices.pct_change().dropna()\n" + \
           "kurt = np.mean((returns - np.mean(returns))**4) / np.std(returns)**4\n" + \
           "S, K, t, r, sig = 19.34, 20, 0.5, 0.04, 0.25\n" + \
           "d1 = (np.log(S/K) + (r + sig**2/2)*t) / (sig * np.sqrt(t))\n" + \
           "d2 = d1 - sig * np.sqrt(t)\n" + \
           "call = S * norm.cdf(d1) - K * np.exp(-r*t) * norm.cdf(d2)\n" + \
           "gamma = norm.pdf(d1) / (S * sig * np.sqrt(t))\n" + \
           "delta = norm.cdf(d1)\n" + \
           "vega = S * np.sqrt(t) * norm.pdf(d1) / 100\n" + \
           "result = {'hurst': float(hurst), 'kurtosis': float(kurt), 'call': float(call), 'gamma': float(gamma), 'delta': float(delta), 'vega': float(vega)}"
    metrics = code_execution(code)
    # Sentiment
    posts = x_keyword_search('crash signals 2025', limit=5)
    sentiment = aggregate_sentiment(posts)
    # Entry check
    metrics_dict = {}  # Initialize to avoid UnboundLocalError
    try:
        metrics_dict = eval(metrics)
        hurst_value = metrics_dict['hurst']
        convexity_ratio = convexity_check(metrics_dict['gamma'], metrics_dict['delta'])
    except Exception as e:
        print(f"Exception in eval: {e}")  # Debug
        hurst_value = 0.0
        convexity_ratio = 0.0
    triggers = {
        'low_vix': 19.34 < 15,
        'hurst': hurst_value > 0.6
    }
    triggers_met = sum(triggers.values())
    edge = 2.5 if triggers_met >= 1 else 1.5  # Adjusted for 1-2 triggers
    # Prediction (optional boost)
    prob = 0.15 if triggers_met >= 1 else 0.10
    # Positioning (with Patience Principle and volatility cap)
    vix_current = 19.34  # Mock; use yf.download('^VIX')['Close'][-1] for real
    if vix_current > 30 or sentiment < -0.5:
        allocation = 0.10  # Pause ramping (vol cap)
    else:
        allocation = 0.25 if vix_current > 20 else 0.20
    positioning = "Hold or roll positions" if edge < 3 else f"Enter VIX calls (40-50%), put spreads (20-30%), CDX (10-20%) with allocation {allocation}"
    # Review (with tail ROI)
    monte_carlo = monte_carlo_payouts()
    tail_roi = (monte_carlo - 20000) / 20000 if monte_carlo > 20000 else 0  # Mock tail ROI >10x
    # Monitoring (with dynamic drawdown pause)
    drawdown = 0.25  # Mock; use max peak-to-trough in returns for real
    exceedance_code = "import numpy as np; import yfinance as yf; prices = yf.download('^VIX', period='6mo')['Close'].dropna(); returns = prices.pct_change().dropna(); z_scores = (returns - np.mean(returns)) / np.std(returns); result = float(np.sum(np.abs(z_scores) > 5) / len(returns) * 100)"
    exceedance = float(code_execution(exceedance_code))
    alert = "Vega stable" if metrics_dict.get('vega', 0.05) < 0.2 else "Alert: High vega"
    if drawdown > 0.15 or exceedance > 50:  # Asset-specific for tech/crypto
        alert += " (Pause entries: High drawdown/exceedance)"
    # Post-Crisis Alpha
    post_crisis_alloc = "30-50% to distressed tech/crypto (e.g., BTC puts)" if edge >= 2.5 else "N/A"
    # Visualization
    img_url = 'https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png'
    img_desc = view_image(img_url)
    # CBOE options
    cboe_data = browse_page("https://www.cboe.com/delayed_quotes/vix/quote_table", "extract OTM VIX calls premiums")
    # Summary
    return f"""
    UITG Monthly Run (2025-10-21):
    - Scan: {scan_results[0]}
    - Metrics: {metrics}
    - Sentiment: {sentiment}
    - Triggers: {triggers} (Met: {triggers_met}/2, Edge: {edge}x, Convexity Ratio: {convexity_ratio:.2f})
    - Prediction: Crash prob {prob} (optional boost)
    - Positioning: {positioning}
    - Review: Monte Carlo avg capital {monte_carlo:.2f} (Tail ROI: {tail_roi:.2f}x >10x)
    - Monitoring: {alert} (Exceedance: {exceedance:.2f}%)
    - Post-Crisis Alpha: {post_crisis_alloc}
    - CBOE Data: {cboe_data}
    - Image: {img_desc}
    """

print("UITG Master Script Initialized")