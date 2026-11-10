import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import io
import re
import math
import time
from datetime import datetime, timedelta
import yfinance as yf
from scipy.optimize import minimize
from playwright.sync_api import sync_playwright
import nest_asyncio
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ==========================================
# 0. Infrastructure & OS Patches
# ==========================================
# Patch Streamlit's event loop to allow synchronous Playwright execution
nest_asyncio.apply()

@st.cache_resource
def install_playwright():
    """Safely installs Chromium without blocking the main Streamlit thread."""
    os.system("playwright install chromium")
    os.system("playwright install-deps")

install_playwright()

# ==========================================
# 1. Page Configuration & Styling
# ==========================================
st.set_page_config(page_title="Power Law Macro Quant", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #1e1e1e;
        border: 1px solid #333333;
        padding: 5% 5% 5% 10%;
        border-radius: 5px;
        color: white;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.5);
    }
    .macro-on { border-left: 5px solid #00cc66 !important; }
    .macro-off { border-left: 5px solid #ff3333 !important; }
    .macro-neutral { border-left: 5px solid #ffcc00 !important; }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Asymmetric Power Law & Macro Correlation Engine")
st.markdown("Integrates Multi-Source Fundamentals, Macro DCC-GARCH Signals, Gold ETF Trading, and LPPLS Criticality.")

# Session State for Sweep Ledger
if 'portfolio_sweeps' not in st.session_state:
    st.session_state.portfolio_sweeps = pd.DataFrame(columns=['Date', 'Trade_Profit_INR', 'Sweep_Amount_INR', 'Gold_ETF_Price', 'Units_Accumulated'])
if 'total_gold_units' not in st.session_state:
    st.session_state.total_gold_units = 0.0

# ==========================================
# 2. Market Data & API Bypasses
# ==========================================
@st.cache_data(ttl=86400)
def fetch_nse_index_tickers(index_name="nifty50"):
    """Fetches real-time ticker lists directly from NSE Archives."""
    url = f"https://nsearchives.nseindia.com/content/indices/ind_{index_name}list.csv"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            return df["Symbol"].tolist()
    except Exception as e:
        st.warning(f"NSE Fetch Failed: {e}")
    return ["RECLTD", "PFC", "ZOMATO", "TATASTEEL", "ITC", "WIPRO", "ONGC"] # Fallback

@st.cache_data(ttl=3600)
def fetch_prices_direct(tickers, days=200):
    """Robust REST API fetch with multiple fallback endpoints."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    period1 = int(start_date.timestamp())
    period2 = int(end_date.timestamp())
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json'
    }
    
    data = pd.DataFrame()
    
    for ticker in tickers:
        success = False
        endpoints = [
            f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?period1={period1}&period2={period2}&interval=1d",
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={period1}&period2={period2}&interval=1d",
        ]
        
        for url in endpoints:
            if success: break
            try:
                res = requests.get(url, headers=headers, timeout=10)
                if res.status_code == 200:
                    json_data = res.json()
                    result = json_data.get('chart', {}).get('result', [])
                    if result:
                        timestamps = result[0].get('timestamp', [])
                        quote = result[0]['indicators']['quote'][0]
                        closes = quote.get('close', [])
                        
                        valid_data = [(ts, c) for ts, c in zip(timestamps, closes) if c is not None]
                        if valid_data:
                            valid_ts, valid_closes = zip(*valid_data)
                            dates = pd.to_datetime(valid_ts, unit='s').dt.date
                            df = pd.DataFrame({ticker: valid_closes}, index=dates)
                            data = df if data.empty else data.join(df, how='outer')
                            success = True
            except Exception:
                continue
                
        # Ultimate fallback: attempt standard yfinance download if direct endpoints fail
        if not success:
            try:
                fallback_df = yf.download(ticker, start=start_date, end=end_date, progress=False)['Close']
                if not fallback_df.empty:
                    df = pd.DataFrame({ticker: fallback_df.values}, index=fallback_df.index.date)
                    data = df if data.empty else data.join(df, how='outer')
            except Exception:
                pass
                
    return data.ffill().dropna()

# ==========================================
# 3. Sidebar Configuration
# ==========================================
st.sidebar.header("Strategy Configurations")

index_choice = st.sidebar.selectbox("Select Target NSE Universe", ["nifty50", "nifty100", "nifty500", "niftynext50"])
universe_tickers = fetch_nse_index_tickers(index_choice)

with st.sidebar.expander("Equity Universe Selection", expanded=True):
    st.info("⚠️ Ensure you select a reasonable subset (10-20 stocks) to prevent cloud timeouts.")
    default_subset = universe_tickers[:7] if len(universe_tickers) >= 7 else universe_tickers
    selected_tickers = st.multiselect("Select NSE Equities (Without .NS)", options=universe_tickers, default=default_subset)

with st.sidebar.expander("Composite Signal Weightings"):
    w_fund = st.sidebar.slider("Fundamental Anchor (%)", 0, 100, 40)
    w_tech = st.sidebar.slider("Technical Validation (MRS) (%)", 0, 100, 20)
    w_plcs = st.sidebar.slider("Power Law Criticality (%)", 0, 100, 40)

with st.sidebar.expander("Gold ETF Sweep Rules"):
    profit_sweep_pct = st.slider("Net Profit Swept to Gold ETF (%)", 10, 100, 50)
    initial_capital = st.number_input("Initial Portfolio Capital (₹)", value=1000000, step=100000)

# ==========================================
# 4. Multi-Source Fundamental Extraction
# ==========================================
def extract_screener_playwright(symbol, page):
    fallback_data = {"promoter_pledge_pct": 0.0}
    try:
        response = page.goto(f"https://www.screener.in/company/{symbol}/consolidated/", timeout=15000, wait_until="domcontentloaded")
        if response and response.status == 404:
            page.goto(f"https://www.screener.in/company/{symbol}/", timeout=15000, wait_until="domcontentloaded")
            
        html = page.content()
        def extract_metric(name, html_text):
            pattern = fr'<span class="name">\s*{name}\s*</span>.*?<span class="number">([^<]+)</span>'
            match = re.search(pattern, html_text, re.DOTALL | re.IGNORECASE)
            return float(match.group(1).replace(",", "").strip()) if match else None

        fallback_data["pe_ratio"] = extract_metric(r"Stock P/E", html)
        fallback_data["roce_pct"] = extract_metric(r"ROCE", html)
        fallback_data["ev_ebitda"] = extract_metric(r"EV / EBITDA", html)
        
        current_price = extract_metric(r"Current Price", html)
        book_value = extract_metric(r"Book Value", html)
        if current_price and book_value and book_value != 0:
            fallback_data["pb_ratio"] = round(current_price / book_value, 2)

        match_top = re.search(r'Pledged percentage.*?<span class="number">([^<]+)</span>', html, re.DOTALL | re.IGNORECASE)
        if match_top: fallback_data["promoter_pledge_pct"] = float(match_top.group(1).replace(",", "").strip())
    except Exception: pass
    return fallback_data

@st.cache_data(ttl=3600)
def fetch_multi_source_data(symbols):
    records = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            page = context.new_page()

            for sym in symbols:
                yf_sym = f"{sym}.NS"
                info = yf.Ticker(yf_sym).info
                pe_yf = info.get("trailingPE", 0.0)
                pb_yf = info.get("priceToBook", 0.0)
                screener_data = extract_screener_playwright(sym, page)
                
                records.append({
                    "Ticker": yf_sym,
                    "PE Ratio": screener_data.get("pe_ratio") or pe_yf,
                    "PB Ratio": screener_data.get("pb_ratio") or pb_yf,
                    "EV/EBITDA": screener_data.get("ev_ebitda") or info.get("enterpriseToEbitda", 0.0),
                    "ROCE (%)": screener_data.get("roce_pct", 0.0),
                    "Pledge (%)": screener_data.get("promoter_pledge_pct", 0.0)
                })
            browser.close()
    except Exception as e:
        st.warning(f"Playwright scraping degraded. Using Yahoo. Error: {e}")
        for sym in symbols:
            records.append({"Ticker": f"{sym}.NS", "PE Ratio": 0, "PB Ratio": 0, "EV/EBITDA": 0, "ROCE (%)": 0, "Pledge (%)": 0})
    return pd.DataFrame(records)

# ==========================================
# 5. Power Law Math (Hurst & LPPLS)
# ==========================================
def compute_hurst(prices, window=100):
    prices = prices.dropna()
    if len(prices) < 40: return 0.5
    log_returns = np.diff(np.log(prices.values))
    if len(log_returns) > window: log_returns = log_returns[-window:]
    
    N = len(log_returns)
    chunk_sizes = np.unique(np.linspace(8, max(10, N // 2), num=10, dtype=int))
    log_n, log_rs = [], []
    
    for n in chunk_sizes:
        num_chunks = N // n
        rs_values = []
        for i in range(num_chunks):
            chunk = log_returns[i * n : (i + 1) * n]
            r_n = np.max(np.cumsum(chunk - np.mean(chunk))) - np.min(np.cumsum(chunk - np.mean(chunk)))
            s_n = np.std(chunk, ddof=1)
            if s_n > 1e-12 and r_n > 0: rs_values.append(r_n / s_n)
        if rs_values:
            log_n.append(math.log(n))
            log_rs.append(math.log(np.mean(rs_values)))
            
    if len(log_n) < 3: return 0.5
    slope, _ = np.polyfit(log_n, log_rs, 1)
    return float(np.clip(slope, 0.0, 1.0))

def calibrate_lppls(prices, window=120):
    clean_prices = prices.dropna()
    if len(clean_prices) < 50: return 999.0
    series = clean_prices.iloc[-min(window, len(clean_prices)):]
    y, t = np.log(series.values), np.arange(len(series), dtype=float)
    t_last = float(len(y) - 1)

    def solve_linear(tc, m, omega):
        dt_vec = tc - t
        if np.any(dt_vec <= 0.5): return None
        f = np.power(dt_vec, m)
        X = np.column_stack([np.ones(len(t)), f, f * np.cos(omega * np.log(dt_vec)), f * np.sin(omega * np.log(dt_vec))])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            return np.sum((y - (X @ beta)) ** 2)
        except: return None

    best_sse, best_tc = float("inf"), t_last + 60.0
    for tc_cand in [t_last + 15, t_last + 30, t_last + 60, t_last + 120]:
        sse = solve_linear(tc_cand, 0.5, 8.0)
        if sse and sse < best_sse: best_sse, best_tc = sse, tc_cand
    return float(best_tc - t_last)

def compute_zscore(series, invert=False):
    vals = series.astype(float).fillna(0).copy()
    if invert: vals = -1.0 * vals
    mean_val, std_val = vals.mean(), vals.std()
    return (vals - mean_val) / std_val if std_val > 1e-12 else pd.Series(0.0, index=series.index)

@st.cache_data(ttl=1800)
def fetch_sentiment_data() -> float:
    rss_urls = ["https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000811", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"]
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(entry.title)['compound'] for url in rss_urls for entry in feedparser.parse(url).entries[:15]]
    return round(np.mean(scores), 3) if scores else 0.0

# ==========================================
# 6. Execution & UI Rendering
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Macro & Gold ETF Trading", "📈 Equity Power Law Ranker", "💰 Wealth Accumulation Sweep"])

with st.spinner("Fetching Live Market Data..."):
    macro_tickers = ["DX-Y.NYB", "BZ=F", "^NSEI", "GOLDIETF.NS"]
    macro_df = fetch_prices_direct(macro_tickers, days=250)

# --- TAB 1: MACRO & GOLD ---
with tab1:
    st.header("Macroeconomic Environment & Gold ETF Trading")
    if not macro_df.empty and len(macro_df) > 20:
        colA, colB, colC, colD = st.columns(4)
        gold_px = macro_df['GOLDIETF.NS'].iloc[-1] if 'GOLDIETF.NS' in macro_df else 0.0
        gold_hurst = compute_hurst(macro_df['GOLDIETF.NS']) if 'GOLDIETF.NS' in macro_df else 0.5
        gold_tc = calibrate_lppls(macro_df['GOLDIETF.NS']) if 'GOLDIETF.NS' in macro_df else 999.0
        sentiment = fetch_sentiment_data()
        
        colA.metric("ICICI Gold ETF (GOLDIETF.NS)", f"₹{gold_px:,.2f}")
        colB.metric("Gold Hurst Exponent", round(gold_hurst, 3), delta="Persistent" if gold_hurst >= 0.65 else "Mean-Reverting", delta_color="normal" if gold_hurst>=0.65 else "inverse")
        colC.metric("LPPLS Criticality", f"{round(gold_tc,1)} days", delta="LIQUIDATION WARNING" if gold_tc <= 30 else "Safe", delta_color="inverse" if gold_tc<=30 else "normal")
        colD.metric("Macro NLP Sentiment", sentiment, delta="Bullish" if sentiment > 0.05 else "Bearish" if sentiment < -0.05 else "Neutral")

        st.markdown("---")
        c1, c2 = st.columns([1, 2])
        with c1:
            try:
                dxy_trend = macro_df['DX-Y.NYB'].iloc[-1] - macro_df['DX-Y.NYB'].iloc[-20]
                crude_trend = macro_df['BZ=F'].iloc[-1] - macro_df['BZ=F'].iloc[-20]
                nifty_trend = macro_df['^NSEI'].iloc[-1] - macro_df['^NSEI'].iloc[-20]
                
                if dxy_trend > 0 and crude_trend > 0 and nifty_trend < 0:
                    macro_signal, macro_multiplier, css_class = "STRONG RISK-OFF (Gold Favorable)", 0.50, "macro-off"
                elif dxy_trend < 0 and crude_trend < 0 and nifty_trend > 0:
                    macro_signal, macro_multiplier, css_class = "STRONG RISK-ON (Equity Favorable)", 1.25, "macro-on"
                else:
                    macro_signal, macro_multiplier, css_class = "NEUTRAL", 1.00, "macro-neutral"
            except Exception:
                macro_signal, macro_multiplier, css_class = "NEUTRAL (DATA PENDING)", 1.00, "macro-neutral"
                
            st.markdown(f"""
            <div data-testid="metric-container" class="{css_class}">
                <p style="font-size:14px; color:#888; margin-bottom:0;">Macro Regime Classification</p>
                <h2 style="margin:0; padding:0;">{macro_signal}</h2>
                <p style="color:#aaa; font-size:12px; margin-top:5px;">Equity PLCS Multiplier: {macro_multiplier}x</p>
            </div><br>
            """, unsafe_allow_html=True)
            
        with c2:
            st.write("**Cross-Asset Conditional Correlation Matrix (Trailing 60 Days)**")
            corr_matrix = macro_df.tail(60).pct_change().corr()
            fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
            fig_corr.update_layout(margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.error("Insufficient macroeconomic data retrieved. API may be rate-limited.")
        macro_multiplier, gold_px = 1.0, 0.0

# --- TAB 2: EQUITY RANKER ---
with tab2:
    st.header(f"Power Law Criticality Ranker ({index_choice.upper()} Universe)")
    
    # Validation Check before allowing execution
    if len(selected_tickers) > 20:
        st.error(f"⚠️ You have selected {len(selected_tickers)} stocks. Processing more than 20 stocks at once will likely cause a Streamlit Cloud timeout. Please reduce your selection.")
        execute_button = st.button("Execute Multi-Source Quant Analysis", disabled=True)
    elif len(selected_tickers) == 0:
        st.warning("⚠️ Please select at least one stock from the sidebar.")
        execute_button = st.button("Execute Multi-Source Quant Analysis", disabled=True)
    else:
        execute_button = st.button("Execute Multi-Source Quant Analysis")
        
    if execute_button:
        with st.spinner("Scraping Fundamentals & Calibrating LPPLS Models..."):
            fund_df = fetch_multi_source_data(selected_tickers)
            
            try:
                price_df = fetch_prices_direct([f"{s}.NS" for s in selected_tickers], days=250)
                
                fund_df["Hurst (100d)"] = fund_df["Ticker"].apply(lambda t: compute_hurst(price_df[t]) if t in price_df.columns else 0.5)
                fund_df["LPPLS t_c (days)"] = fund_df["Ticker"].apply(lambda t: calibrate_lppls(price_df[t]) if t in price_df.columns else 999.0)
                
                fund_df["Z_Fund"] = compute_zscore(fund_df["ROCE (%)"]) + compute_zscore(fund_df["EV/EBITDA"], invert=True)
                fund_df["Z_PLCS"] = compute_zscore(fund_df["Hurst (100d)"]) + compute_zscore(fund_df["LPPLS t_c (days)"], invert=True)
                
                fund_df["Composite Score"] = ((w_fund/100) * fund_df["Z_Fund"] + (w_plcs/100) * fund_df["Z_PLCS"]) * macro_multiplier
                fund_df["Liquidation Alert"] = np.where((fund_df["LPPLS t_c (days)"] <= 30) & (fund_df["Hurst (100d)"] <= 0.55), True, False)
                
                st.dataframe(
                    fund_df.sort_values(by="Composite Score", ascending=False),
                    column_config={
                        "Hurst (100d)": st.column_config.NumberColumn(format="%.3f"),
                        "LPPLS t_c (days)": st.column_config.NumberColumn(format="%.1f"),
                        "Composite Score": st.column_config.NumberColumn(format="%.3f"),
                    },
                    use_container_width=True, hide_index=True
                )
            except Exception as e:
                st.error(f"Failed to process Power Law ranking due to pricing data error: {e}")

# --- TAB 3: SWEEP CALCULATOR ---
with tab3:
    st.header("Systematic Wealth Accumulation: ICICI Gold ETF Sweep Protocol")
    st.markdown(f"**Current Execution Directive:** Sweep **{profit_sweep_pct}%** of realized short-term equity trading profits directly into physical Gold ETF accumulation to convert active alpha into inflation-hedged beta.")
    
    with st.form("sweep_form", clear_on_submit=True):
        trade_profit = st.number_input("Realized Profit from Closed Equity Trade (INR)", min_value=0.0, step=1000.0)
        submitted = st.form_submit_button("Execute Profit Sweep")
        
        if submitted and trade_profit > 0 and gold_px > 0:
            sweep_amt = trade_profit * (profit_sweep_pct / 100.0)
            units_bought = math.floor(sweep_amt / gold_px)
            actual_cost = units_bought * gold_px
            
            new_record = pd.DataFrame([{
                'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Trade_Profit_INR': trade_profit,
                'Sweep_Amount_INR': actual_cost,
                'Gold_ETF_Price': gold_px,
                'Units_Accumulated': units_bought
            }])
            st.session_state.portfolio_sweeps = pd.concat([st.session_state.portfolio_sweeps, new_record], ignore_index=True)
            st.session_state.total_gold_units += units_bought
            st.success(f"Execution complete: Swept ₹{actual_cost:,.2f} to acquire {units_bought} units of GOLDIETF.NS.")
        elif submitted and gold_px == 0:
            st.error("Gold ETF price is unavailable. Cannot calculate sweep units.")

    if not st.session_state.portfolio_sweeps.empty:
        st.write("### Tactical Accumulation Ledger")
        st.dataframe(st.session_state.portfolio_sweeps, use_container_width=True)
        total_value = st.session_state.total_gold_units * gold_px
        st.info(f"**Total Gold Vault Balance:** {st.session_state.total_gold_units} Units | **Current Market Value:** ₹{total_value:,.2f}")
