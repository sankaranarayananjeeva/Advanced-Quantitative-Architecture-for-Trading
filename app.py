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
nest_asyncio.apply()

@st.cache_resource
def install_playwright():
    """Safely installs Chromium without blocking the main Streamlit thread."""
    try:
        os.system("playwright install chromium")
        os.system("playwright install-deps")
    except:
        pass

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
    .pred-box {
        background-color: #2b2b2b;
        border: 1px solid #444;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
    }
    .pred-high { color: #00cc66; font-size: 20px; font-weight: bold; }
    .pred-low { color: #ff3333; font-size: 20px; font-weight: bold; }
    .pred-neutral { color: #ffcc00; font-size: 16px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Asymmetric Power Law & Macro Correlation Engine")

# ==========================================
# Session State Initialization (For Background Queue & History)
# ==========================================
if 'portfolio_sweeps' not in st.session_state:
    st.session_state.portfolio_sweeps = pd.DataFrame(columns=['Date', 'Gross_Equity_Profit', 'Net_Post_Tax_Profit', 'Sweep_Amount_INR', 'Gold_ETF_Price', 'Units_Accumulated'])
if 'total_gold_units' not in st.session_state:
    st.session_state.total_gold_units = 0.0

if "bg_queue" not in st.session_state:
    st.session_state.bg_queue = []
if "bg_results" not in st.session_state:
    st.session_state.bg_results = pd.DataFrame()
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "total_to_process" not in st.session_state:
    st.session_state.total_to_process = 0
    
# NEW: Array to store the last 5 executed runs
if "run_history" not in st.session_state:
    st.session_state.run_history = []

# ==========================================
# 2. Universal Resilient YFinance Fetcher
# ==========================================
def robust_yf_fetch(ticker, days=250):
    for attempt in range(4):
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=f"{days}d")
            if not df.empty and 'Close' in df.columns:
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None).normalize()
                return df[['Open', 'High', 'Low', 'Close']]
        except Exception:
            pass
        
        time.sleep(1.5 ** attempt) 
        
        try:
            end = datetime.today()
            start = end - timedelta(days=days)
            df = yf.download(ticker, start=start, end=end, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None).normalize()
                if 'Close' in df.columns:
                    return df[['Open', 'High', 'Low', 'Close']]
        except Exception:
            pass
            
        time.sleep(1.5 ** attempt)
        
    return pd.DataFrame()

@st.cache_data(ttl=86400)
def fetch_nse_index_tickers(index_name="nifty50"):
    url = f"https://nsearchives.nseindia.com/content/indices/ind_{index_name}list.csv"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        if r.status_code == 200:
            return pd.read_csv(io.StringIO(r.text))["Symbol"].tolist()
    except Exception:
        pass
    return ["RECLTD", "PFC", "ZOMATO", "TATASTEEL", "ITC", "WIPRO", "ONGC"] 

@st.cache_data(ttl=3600)
def fetch_macro_data(days=250):
    tickers = ["DX-Y.NYB", "BZ=F", "^NSEI", "GOLDIETF.NS", "INR=X", "EURUSD=X", "GBPUSD=X", "JPY=X", "GC=F"]
    
    macro_close_dict = {}
    ohlc_dict = {}
    
    for t in tickers:
        df = robust_yf_fetch(t, days)
        if not df.empty:
            macro_close_dict[t] = df['Close']
            ohlc_dict[t] = df
        time.sleep(1.5) 
                
    if macro_close_dict:
        merged_close = pd.DataFrame(macro_close_dict).ffill().dropna()
        return merged_close, ohlc_dict
    return pd.DataFrame(), {}

# ==========================================
# 3. Garman-Klass & Error Correction Model
# ==========================================
def calculate_advanced_gold_bounds(macro_df, ohlc_dict):
    if 'GOLDIETF.NS' not in ohlc_dict or 'GC=F' not in macro_df or 'INR=X' not in macro_df:
        return None

    gold_df = ohlc_dict['GOLDIETF.NS'].tail(30).copy()
    gold_df = gold_df[(gold_df['High'] > 0) & (gold_df['Low'] > 0) & (gold_df['Open'] > 0) & (gold_df['Close'] > 0)]
    
    if len(gold_df) < 14:
        return None
        
    gold_df['log_hl'] = np.log(gold_df['High'] / gold_df['Low'])
    gold_df['log_co'] = np.log(gold_df['Close'] / gold_df['Open'])
    
    gold_df['gk_var'] = 0.5 * (gold_df['log_hl'] ** 2) - (2 * np.log(2) - 1) * (gold_df['log_co'] ** 2)
    gold_df['gk_var'] = gold_df['gk_var'].clip(lower=0.000001)
    gk_daily_vol = np.sqrt(gold_df['gk_var'].rolling(14).mean().iloc[-1])
    
    recent_macro = macro_df.tail(60).copy()
    y = np.log(recent_macro['GOLDIETF.NS'].values)
    theoretical_int = (recent_macro['GC=F'].values / 31.1034768) * recent_macro['INR=X'].values
    x = np.log(theoretical_int)
    
    beta, alpha = np.polyfit(x, y, 1)
    
    spread = y - (beta * x + alpha)
    current_spread = spread[-1]
    spread_mean = np.mean(spread)
    spread_std = np.std(spread)
    
    z_spread = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0
    error_correction_bias = -0.5 * z_spread * gk_daily_vol 
    
    last_close = gold_df['Close'].iloc[-1]
    
    predicted_mid = last_close * (1 + error_correction_bias)
    breakout_buy = predicted_mid + (last_close * gk_daily_vol * 1.5)
    dynamic_stop = predicted_mid - (last_close * gk_daily_vol * 1.5)

    return {
        "last_close": last_close,
        "gk_vol_pct": gk_daily_vol * 100,
        "z_spread": z_spread,
        "breakout_buy": breakout_buy,
        "dynamic_stop": dynamic_stop,
        "sebi_upper": last_close * 1.06,
        "sebi_lower": last_close * 0.94
    }

# ==========================================
# 4. Sidebar Configuration
# ==========================================
st.sidebar.header("Strategy Configurations")

index_choice = st.sidebar.selectbox("Select Target NSE Universe", ["nifty50", "nifty100", "nifty500", "niftynext50"])
universe_tickers = fetch_nse_index_tickers(index_choice)

with st.sidebar.expander("Equity Universe Selection", expanded=True):
    st.info("💡 Background Queue Enabled: You can process up to 500 stocks without timeouts.")
    selected_tickers = st.multiselect("Select NSE Equities (Without .NS)", options=universe_tickers, default=universe_tickers[:10])

with st.sidebar.expander("Composite Signal Weightings"):
    w_fund = st.sidebar.slider("Fundamental Anchor (%)", 0, 100, 40)
    w_tech = st.sidebar.slider("Technical Validation (MRS) (%)", 0, 100, 20)
    w_plcs = st.sidebar.slider("Power Law Criticality (%)", 0, 100, 40)

with st.sidebar.expander("Gold ETF Sweep Rules"):
    profit_sweep_pct = st.slider("Net Profit Swept to Gold ETF (%)", 10, 100, 50)
    initial_capital = st.number_input("Initial Portfolio Capital (₹)", value=1000000, step=100000)

# ==========================================
# 5. Power Law Math & Multi-Source Scraper
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
        
        match_top = re.search(r'Pledged percentage.*?<span class="number">([^<]+)</span>', html, re.DOTALL | re.IGNORECASE)
        if match_top: fallback_data["promoter_pledge_pct"] = float(match_top.group(1).replace(",", "").strip())
    except Exception: pass
    return fallback_data

def process_symbol_chunk(symbols):
    """Processes a chunk of symbols sequentially to avoid 429 timeouts."""
    records = []
    
    price_data = {}
    for sym in symbols:
        df = robust_yf_fetch(f"{sym}.NS", 200)
        if not df.empty:
            price_data[f"{sym}.NS"] = df['Close']
        time.sleep(1.0) 

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent="Mozilla/5.0")
            page = context.new_page()

            for sym in symbols:
                yf_sym = f"{sym}.NS"
                try:
                    pe_yf, ev_ebitda_yf = 0.0, 0.0
                    try:
                        info = yf.Ticker(yf_sym).info
                        pe_yf = info.get("trailingPE", 0.0)
                        ev_ebitda_yf = info.get("enterpriseToEbitda", 0.0)
                    except: pass
                    
                    screener_data = extract_screener_playwright(sym, page)
                    
                    px_data = price_data.get(yf_sym, pd.Series()).dropna()
                    hurst_val = compute_hurst(px_data) if len(px_data) > 40 else 0.5
                    tc_val = calibrate_lppls(px_data) if len(px_data) > 50 else 999.0
                    
                    records.append({
                        "Ticker": yf_sym,
                        "PE Ratio": screener_data.get("pe_ratio") or pe_yf,
                        "EV/EBITDA": screener_data.get("ev_ebitda") or ev_ebitda_yf,
                        "ROCE (%)": screener_data.get("roce_pct", 0.0),
                        "Pledge (%)": screener_data.get("promoter_pledge_pct", 0.0),
                        "Hurst (100d)": hurst_val,
                        "LPPLS t_c (days)": tc_val
                    })
                except Exception as e:
                    pass
            browser.close()
    except Exception:
        pass
        
    return pd.DataFrame(records)

def compute_hurst(prices, window=100):
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
    series = prices.iloc[-min(window, len(prices)):]
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
    try:
        rss_urls = ["https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000811", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"]
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(entry.title)['compound'] for url in rss_urls for entry in feedparser.parse(url).entries[:15]]
        return round(np.mean(scores), 3) if scores else 0.0
    except Exception:
        return 0.0

# ==========================================
# 6. UI Rendering & Background Queue Processor
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Macro & Gold ETF Trading", "📈 Background Equity Ranker", "💰 Wealth Accumulation Sweep"])

with st.spinner("Fetching Market Data sequentially (Bypassing API Rate Limits)..."):
    macro_df, ohlc_dict = fetch_macro_data(days=250)
    advanced_bounds = calculate_advanced_gold_bounds(macro_df, ohlc_dict)

# --- TAB 1: MACRO & GOLD ---
with tab1:
    st.header("Macroeconomic Environment & Gold ETF Execution")
    
    if not macro_df.empty and 'GOLDIETF.NS' in macro_df.columns and advanced_bounds:
        colA, colB, colC, colD = st.columns(4)
        gold_px = macro_df['GOLDIETF.NS'].iloc[-1]
        gold_hurst = compute_hurst(macro_df['GOLDIETF.NS'])
        gold_tc = calibrate_lppls(macro_df['GOLDIETF.NS'])
        sentiment = fetch_sentiment_data()
        
        colA.metric("ICICI Gold ETF (GOLDIETF.NS)", f"₹{gold_px:,.2f}")
        colB.metric("Gold Hurst Exponent", round(gold_hurst, 3), delta="Persistent" if gold_hurst >= 0.65 else "Mean-Reverting", delta_color="normal" if gold_hurst>=0.65 else "inverse")
        colC.metric("LPPLS Criticality", f"{round(gold_tc,1)} days", delta="LIQUIDATION WARNING" if gold_tc <= 30 else "Safe", delta_color="inverse" if gold_tc<=30 else "normal")
        colD.metric("Macro NLP Sentiment", sentiment, delta="Bullish" if sentiment > 0.05 else "Bearish" if sentiment < -0.05 else "Neutral")

        # Predictive Execution Box (Garman-Klass + Error Correction Spread)
        st.markdown(f"""
        <div class="pred-box">
            <p style="margin:0; color:#aaa; font-size:14px;">Next Session Algorithmic Execution Triggers (Garman-Klass Volatility + VECM Spread)</p>
            <div style="display:flex; justify-content:space-around; align-items:center; margin-top:10px;">
                <div>
                    <span style="color:#aaa; font-size:12px;">Dynamic Stop Loss (-1.5x GK)</span><br>
                    <span class="pred-low">₹{advanced_bounds['dynamic_stop']:.2f}</span>
                    <br><span style="color:#666; font-size:10px;">SEBI Lower Limit: ₹{advanced_bounds['sebi_lower']:.2f}</span>
                </div>
                <div>
                    <span class="pred-neutral">Daily GK Volatility: {advanced_bounds['gk_vol_pct']:.2f}%</span><br>
                    <span style="color:#888; font-size:11px;">(Z-Score Spread vs Int. Gold: {advanced_bounds['z_spread']:+.2f})</span>
                </div>
                <div>
                    <span style="color:#aaa; font-size:12px;">Breakout Buy Trigger (+1.5x GK)</span><br>
                    <span class="pred-high">₹{advanced_bounds['breakout_buy']:.2f}</span>
                    <br><span style="color:#666; font-size:10px;">SEBI Upper Limit: ₹{advanced_bounds['sebi_upper']:.2f}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")

        st.markdown("---")
        c1, c2 = st.columns([1, 2])
        with c1:
            dxy_trend, crude_trend, nifty_trend = 0, 0, 0
            if 'DX-Y.NYB' in macro_df.columns and len(macro_df) > 20: dxy_trend = macro_df['DX-Y.NYB'].iloc[-1] - macro_df['DX-Y.NYB'].iloc[-20]
            if 'BZ=F' in macro_df.columns and len(macro_df) > 20: crude_trend = macro_df['BZ=F'].iloc[-1] - macro_df['BZ=F'].iloc[-20]
            if '^NSEI' in macro_df.columns and len(macro_df) > 20: nifty_trend = macro_df['^NSEI'].iloc[-1] - macro_df['^NSEI'].iloc[-20]
                
            if dxy_trend > 0 and crude_trend > 0 and nifty_trend < 0:
                macro_signal, macro_multiplier, css_class = "STRONG RISK-OFF (Gold Favorable)", 0.50, "macro-off"
            elif dxy_trend < 0 and crude_trend < 0 and nifty_trend > 0:
                macro_signal, macro_multiplier, css_class = "STRONG RISK-ON (Equity Favorable)", 1.25, "macro-on"
            else:
                macro_signal, macro_multiplier, css_class = "NEUTRAL", 1.00, "macro-neutral"
                
            st.markdown(f"""
            <div data-testid="metric-container" class="{css_class}">
                <p style="font-size:14px; color:#888; margin-bottom:0;">Macro Regime Classification</p>
                <h2 style="margin:0; padding:0;">{macro_signal}</h2>
                <p style="color:#aaa; font-size:12px; margin-top:5px;">Equity PLCS Multiplier: {macro_multiplier}x</p>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            if len(macro_df) > 60:
                st.write("**Cross-Asset Correlation Matrix (Inc. Major Currencies)**")
                corr_matrix = macro_df.tail(60).pct_change().corr()
                fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
                fig_corr.update_layout(margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.error("Insufficient macroeconomic data retrieved. API may be rate-limited.")
        macro_multiplier, gold_px = 1.0, 0.0

# --- TAB 2: BACKGROUND EQUITY RANKER ---
with tab2:
    st.header(f"Background Power Law Ranker ({index_choice.upper()} Universe)")
    st.write("Iterative background queue processing to bypass Streamlit Cloud timeouts.")
    
    col_btn, col_prog = st.columns([1, 3])
    with col_btn:
        if st.button("Start Background Execution", disabled=st.session_state.is_processing):
            if len(selected_tickers) > 0:
                st.session_state.bg_queue = selected_tickers.copy()
                st.session_state.total_to_process = len(selected_tickers)
                st.session_state.bg_results = pd.DataFrame()
                st.session_state.is_processing = True
                st.rerun()
            else:
                st.error("Please select at least 1 ticker.")
    
    with col_prog:
        if st.button("Stop/Clear Queue"):
            st.session_state.is_processing = False
            st.session_state.bg_queue = []
            st.rerun()

    if st.session_state.is_processing:
        if len(st.session_state.bg_queue) > 0:
            chunk = st.session_state.bg_queue[:5]
            st.session_state.bg_queue = st.session_state.bg_queue[5:]
            
            processed_count = st.session_state.total_to_process - len(st.session_state.bg_queue)
            progress_pct = processed_count / st.session_state.total_to_process
            st.progress(progress_pct, text=f"Processing Chunk: {chunk} ({processed_count}/{st.session_state.total_to_process})")
            
            chunk_df = process_symbol_chunk(chunk)
            
            if not chunk_df.empty:
                st.session_state.bg_results = pd.concat([st.session_state.bg_results, chunk_df], ignore_index=True)
                
            # NEW: Store finalized dataframe to history when the queue finishes
            if len(st.session_state.bg_queue) == 0:
                st.session_state.is_processing = False
                
                if not st.session_state.bg_results.empty:
                    fund_df = st.session_state.bg_results.copy()
                    fund_df["Z_Fund"] = compute_zscore(fund_df["ROCE (%)"]) + compute_zscore(fund_df["EV/EBITDA"], invert=True)
                    fund_df["Z_PLCS"] = compute_zscore(fund_df["Hurst (100d)"]) + compute_zscore(fund_df["LPPLS t_c (days)"], invert=True)
                    fund_df["Composite Score"] = ((w_fund/100) * fund_df["Z_Fund"] + (w_plcs/100) * fund_df["Z_PLCS"]) * macro_multiplier
                    fund_df["Liquidation Alert"] = np.where((fund_df["LPPLS t_c (days)"] <= 30) & (fund_df["Hurst (100d)"] <= 0.55), True, False)
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.run_history.append({"time": timestamp, "data": fund_df})
                    
                    # Enforce a maximum of 5 historical runs in memory
                    if len(st.session_state.run_history) > 5:
                        st.session_state.run_history = st.session_state.run_history[-5:]
                        
                    # Save a local CSV backup
                    os.makedirs("history", exist_ok=True)
                    safe_time = timestamp.replace(":", "-").replace(" ", "_")
                    fund_df.to_csv(f"history/run_{safe_time}.csv", index=False)
            
            time.sleep(1.0) 
            st.rerun()
            
        else:
            st.session_state.is_processing = False
            st.success("Background Execution Complete!")

    # Display Current Run (if available and not processing)
    if not st.session_state.bg_results.empty and not st.session_state.is_processing:
        st.write("### Latest Computed Rankings")
        latest_df = st.session_state.run_history[-1]["data"] if st.session_state.run_history else pd.DataFrame()
        if not latest_df.empty:
            st.dataframe(
                latest_df.sort_values(by="Composite Score", ascending=False),
                column_config={
                    "Hurst (100d)": st.column_config.NumberColumn(format="%.3f"),
                    "LPPLS t_c (days)": st.column_config.NumberColumn(format="%.1f"),
                    "Composite Score": st.column_config.NumberColumn(format="%.3f"),
                },
                use_container_width=True, hide_index=True
            )

    # NEW: Run History Dropdown Feature
    if len(st.session_state.run_history) > 0 and not st.session_state.is_processing:
        st.markdown("---")
        st.subheader("🕰️ Historical Runs (Last 5)")
        history_options = {f"Run at {run['time']} ({len(run['data'])} stocks)": run['data'] for run in reversed(st.session_state.run_history)}
        selected_hist = st.selectbox("Select a previous run to view:", list(history_options.keys()))
        
        if selected_hist:
            hist_df = history_options[selected_hist]
            st.dataframe(
                hist_df.sort_values(by="Composite Score", ascending=False),
                column_config={
                    "Hurst (100d)": st.column_config.NumberColumn(format="%.3f"),
                    "LPPLS t_c (days)": st.column_config.NumberColumn(format="%.1f"),
                    "Composite Score": st.column_config.NumberColumn(format="%.3f"),
                },
                use_container_width=True, hide_index=True
            )

# --- TAB 3: SWEEP CALCULATOR ---
with tab3:
    st.header("Wealth Accumulation: Gold ETF Sweep Protocol (Union Budget 2024 Rules)")
    st.markdown(f"**Execution Directive:** Sweep **{profit_sweep_pct}%** of realized short-term equity profits into physical Gold ETF accumulation.")
    
    with st.form("sweep_form", clear_on_submit=True):
        trade_profit = st.number_input("Gross Profit from Closed Equity Trade (INR)", min_value=0.0, step=1000.0)
        submitted = st.form_submit_button("Execute Profit Sweep")
        
        if submitted and trade_profit > 0 and gold_px > 0:
            net_equity_profit = trade_profit * 0.80  # Subtracting 20% STCG
            sweep_amt = net_equity_profit * (profit_sweep_pct / 100.0)
            units_bought = math.floor(sweep_amt / gold_px)
            actual_cost = units_bought * gold_px
            
            new_record = pd.DataFrame([{
                'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Gross_Equity_Profit': trade_profit,
                'Net_Post_Tax_Profit': net_equity_profit,
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
        gold_ltcg = total_value * 0.125 # 12.5% LTCG assumption on exit without indexation
        net_liquid_value = total_value - gold_ltcg
        
        st.info(f"**Total Gold Vault Balance:** {st.session_state.total_gold_units} Units | **Gross Market Value:** ₹{total_value:,.2f} | **Net Liquid Value (After 12.5% LTCG):** ₹{net_liquid_value:,.2f}")