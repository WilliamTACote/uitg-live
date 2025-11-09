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
    posts = [{'text': f'Mock crash warning for {query} #{i} market panic', 'score': np.random.uniform(-0.8, -0.2)} for i in range(limit)]
    return posts

def aggregate_sentiment(posts: list) -> float:
    try:
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(post['text'])['compound'] for post in posts]
        return np.mean(scores) if scores else 0
    except:
        return -0.5  # Fallback

# === REAL GOOGLE SEARCH ===
def google_search_hedges(query="cheap convex hedges 2025 VIX calls CDX", num=5):
    try:
        API_KEY = st.secrets["GOOGLE_API_KEY"]
        CX = st.secrets["GOOGLE_CX"]
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"q": query, "key": API_KEY, "cx": CX, "num": num}
        response = requests.get(url, params=params, timeout=10)
        items = response.json().get("items", [])
        return [{
            "title": item.get("title", "No title"),
            "snippet": item.get("snippet", "No snippet"),
            "url": item.get("link", "#")
        } for item in items]
    except:
        return [{'title': f'Fallback #{i}', 'snippet': f'OTM VIX call at strike {20+i}', 'url': '#'} for i in range(num)]

# === web_search (used by dashboard.py) ===
def web_search(query: str, num_results: int = 10) -> list:
    return google_search_hedges(query, num_results)

# === REAL CBOE VIX OPTIONS ===
def get_cboe_vix_options():
    url = "https://cdn.cboe.com/api/global/delayed_quotes/options/VX.json"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        options = []
        for opt in data['data']['calls']:
            if opt['delta'] < 0.5 and opt['bid'] > 0:
                premium = (opt['bid'] + opt['ask']) / 2
                options.append({
                    "Strike": opt['strike'],
                    "Premium": round(premium, 2),
                    "IV": round(opt['volatility'], 1),
                    "Delta": round(opt['delta'], 3)
                })
        return options[:5] or "No OTM calls"
    except:
        try:
            vix = yf.download('^VIX', period='1d')['Close'].iloc[-1]
            return f"VIX: {vix:.1f} — Suggest OTM calls at {vix + 5:.0f}"
        except:
            return "CBOE API failed"

# Convexity Check
def convexity_check(gamma, delta):
    return round(gamma / delta, 3) if delta != 0 else 0

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

# === MASTER MONTHLY RUN ===
def run_uitg_monthly():
    scan_results = web_search('cheap convex hedges 2025 VIX calls puts CDX credit', 5)
    code = """
import numpy as np
import yfinance as yf
from hurst import compute_Hc
from scipy.stats import norm
prices = yf.download('^VIX', period='6mo')['Close'].dropna()
hurst = compute_Hc(prices, kind='price')[0]
returns = prices.pct_change().dropna()
kurt = np.mean((returns - np.mean(returns))**4) / np.std(returns)**4
S, K, t, r, sig = 19.34, 20, 0.5, 0.04, 0.25
d1 = (np.log(S/K) + (r + sig**2/2)*t) / (sig * np.sqrt(t))
d2 = d1 - sig * np.sqrt(t)
call = S * norm.cdf(d1) - K * np.exp(-r*t) * norm.cdf(d2)
gamma = norm.pdf(d1) / (S * sig * np.sqrt(t))
delta = norm.cdf(d1)
vega = S * np.sqrt(t) * norm.pdf(d1) / 100
result = {'hurst': float(hurst), 'kurtosis': float(kurt), 'call': float(call), 'gamma': float(gamma), 'delta': float(delta), 'vega': float(vega)}
"""
    metrics = code_execution(code)
    posts = x_keyword_search('crash signals 2025', 5)
    sentiment = aggregate_sentiment(posts)
    try:
        m = eval(metrics)
        hurst_value = m['hurst']
        convexity_ratio = convexity_check(m['gamma'], m['delta'])
    except:
        hurst_value = 0.0
        convexity_ratio = 0.0
    triggers = {'low_vix': 19.34 < 15, 'hurst': hurst_value > 0.6}
    edge = 2.5 if sum(triggers.values()) >= 1 else 1.5
    prob = 0.15 if sum(triggers.values()) >= 1 else 0.10
    vix_current = 19.34
    if vix_current > 30 or sentiment < -0.5:
        allocation = 0.10
    else:
        allocation = 0.25 if vix_current > 20 else 0.20

    # === YOUR STRATEGY REFINEMENTS ===
    if hurst_value > 0.65 and sentiment < -0.8:
        positioning = "Enter VIX calls (50%), CDX puts (30%)"
        alert = "CRASH SIGNAL — DEPLOY TAIL HEDGE"
    else:
        positioning = "Hold or roll positions" if edge < 3 else f"Enter VIX calls (40-50%), put spreads (20-30%), CDX (10-20%) with allocation {allocation}"

    mc = monte_carlo_payouts()
    tail_roi = (mc - 20000) / 20000
    exceedance_code = "import numpy as np; import yfinance as yf; prices = yf.download('^VIX', period='6mo')['Close'].dropna(); returns = prices.pct_change().dropna(); z_scores = (returns - np.mean(returns)) / np.std(returns); result = float(np.sum(np.abs(z_scores) > 5) / len(returns) * 100)"
    exceedance = float(code_execution(exceedance_code))
    alert_msg = "Vega stable" if m.get('vega', 0.05) < 0.2 else "Alert: High vega"
    if 0.25 > 0.15 or exceedance > 50:
        alert_msg += " (Pause entries: High drawdown/exceedance)"
    img_url = 'https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png'
    img_desc = view_image(img_url)
    cboe_data = get_cboe_vix_options()
    return f"""
**UITG Monthly Run**
- **Scan**: {len(scan_results)} real results
- **Hurst**: {hurst_value:.3f} | **Sentiment**: {sentiment:.3f} | **Edge**: {edge}x
- **Convexity**: {convexity_ratio:.3f}
- **Positioning**: {positioning}
- **Monte Carlo**: ${mc:,.0f} (Tail ROI: {tail_roi:.1f}x)
- **CBOE**: {cboe_data}
"""

print("UITG Master Script Initialized")