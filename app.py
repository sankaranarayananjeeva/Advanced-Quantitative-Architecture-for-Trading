"""
Advanced Quantitative Architecture for Gold ETF Trading
=======================================================
Features:
1. YFinance Market Data Ingestion (ICICI Gold ETF, DXY, USD/INR, Crude)
2. Macroeconomic Correlation Matrix (DCC-GARCH proxy via rolling windows)
3. VADER NLP Sentiment Analysis on RSS Feeds (Bloomberg, Reuters)
4. Multi-Timeframe Power Law Criticality (Hurst & LPPLS with Scipy)
5. Profit Sweep Accumulation Calculator (Converting Alpha to Beta)
6. State-Managed Streamlit User Interface
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import math
import datetime as dt

# ---------------------------------------------------------------------------
# Streamlit Page Configuration & Session State
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Gold ETF Power Law Framework", layout="wide", initial_sidebar_state="expanded")

# Utilize session_state to prevent data loss on widget interaction reruns
if 'portfolio_sweeps' not in st.session_state:
    st.session_state.portfolio_sweeps = pd.DataFrame(columns=['Date', 'Trade_Profit_INR', 'Sweep_Amount_INR', 'Gold_ETF_Price', 'Units_Accumulated'])
if 'total_gold_units' not in st.session_state:
    st.session_state.total_gold_units = 0.0

# ---------------------------------------------------------------------------
# 1. Market Data Ingestion Module (yfinance)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_macro_data(lookback_days: int = 750) -> pd.DataFrame:
    """
    Fetches daily closing prices for ICICI Gold ETF and Macro drivers.
    GOLDIETF.NS : ICICI Prudential Gold ETF
    DX-Y.NYB    : US Dollar Index (Inverse correlation tracking)
    INR=X       : USD/INR Exchange Rate
    CL=F        : Brent Crude Oil Futures (Inflation proxy)
    ^NSEI       : Nifty 50 Index (Risk-On/Risk-Off proxy)
    """
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=lookback_days)
    tickers = ["GOLDIETF.NS", "DX-Y.NYB", "INR=X", "CL=F", "^NSEI"]
    
    # Supress progress bar for clean server logs
    data = yf.download(tickers, start=start_date.isoformat(), end=end_date.isoformat(), progress=False)['Close']
    data = data.ffill().dropna()
    return data

# ---------------------------------------------------------------------------
# 2. Power Law Criticality: Hurst Exponent (Fractal Persistence)
# ---------------------------------------------------------------------------
def compute_hurst(prices: pd.Series, window: int = 100) -> float:
    """Calculates the Hurst Exponent using Rescaled Range (R/S) analysis."""
    if len(prices) < window: return np.nan
    log_rets = np.diff(np.log(prices.values))
    
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(log_rets[lag:], log_rets[:-lag]))) for lag in lags]
    
    # Linear fit to log-log plot to extract the scaling exponent
    m = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = m[0] * 2.0
    return float(np.clip(hurst, 0.0, 1.0))

# ---------------------------------------------------------------------------
# 3. Power Law Criticality: LPPLS Calibration (Filimonov-Sornette)
# ---------------------------------------------------------------------------
def calibrate_lppls(prices: pd.Series) -> dict:
    """Calibrates LPPLS non-linear parameters to detect critical singularity (t_c)."""
    y = np.log(prices.values)
    t = np.arange(len(y))
    t_last = float(len(y) - 1)
    
    def objective_func(params):
        tc, m, w = params
        dt_vec = tc - t
        if np.any(dt_vec <= 0.5): return 1e9
        
        # Slaving transformation matrices
        f = np.power(dt_vec, m)
        g = f * np.cos(w * np.log(dt_vec))
        h = f * np.sin(w * np.log(dt_vec))
        
        X = np.column_stack([np.ones(len(t)), f, g, h])
        try:
            # OLS optimization for linear parameters A, B, C1, C2
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            y_hat = X @ beta
            sse = np.sum((y - y_hat) ** 2)
            # Soft penalty to encourage B < 0 (ensuring an upward positive bubble trajectory)
            penalty = 0 if beta[1] < 0 else (beta[1]**2) * 10
            return sse + penalty
        except np.linalg.LinAlgError:
            return 1e9

    # Strict physical bounds for optimization
    bounds = [(t_last + 5.0, t_last + 150.0), (0.1, 0.9), (6.0, 13.0)]
    initial_guess = [t_last + 30.0, 0.5, 8.0]
    
    # Utilize L-BFGS-B algorithm for bounded non-linear optimization
    opt = minimize(objective_func, x0=initial_guess, bounds=bounds, method='L-BFGS-B')
    tc_opt, m_opt, w_opt = opt.x
    
    tc_days_ahead = round(float(tc_opt - t_last), 1)
    # Trigger liquidation warning if singularity is <= 30 trading days
    return {"tc_days_ahead": tc_days_ahead, "m": m_opt, "omega": w_opt, "exit_warning": tc_days_ahead <= 30.0}

# ---------------------------------------------------------------------------
# 4. NLP Sentiment Analysis Module (VADER + RSS Pipeline)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_sentiment_data() -> float:
    """Fetches RSS feeds for Gold and macro news, scoring via VADER Lexicon."""
    rss_urls = [
        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000811", # CNBC Gold
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml" # WSJ Markets
    ]
    
    analyzer = SentimentIntensityAnalyzer()
    compound_scores = []
    
    for url in rss_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries[:15]: # Process top 15 recent articles per source
            # VADER is highly optimized for short headline text processing
            score = analyzer.polarity_scores(entry.title)['compound']
            compound_scores.append(score)
            
    avg_sentiment = np.mean(compound_scores) if compound_scores else 0.0
    return round(avg_sentiment, 3)

# ---------------------------------------------------------------------------
# 5. Streamlit Application & Dashboard UI
# ---------------------------------------------------------------------------
def main():
    st.title("🏛️ Gold ETF Power Law Trading Framework")
    st.markdown("Advanced Multi-Factor Architecture incorporating Macro Correlations, LPPLS Criticality, and NLP Sentiment.")
    
    # --- Sidebar Configuration ---
    st.sidebar.header("System Configuration")
    lppls_window = st.sidebar.slider("LPPLS Lookback Window", 60, 252, 120, help="Days used to calibrate the singularity model.")
    hurst_window = st.sidebar.slider("Hurst Lookback Window", 50, 200, 100, help="Days to measure fractal persistence.")
    sweep_pct = st.sidebar.slider("Profit Sweep Percentage (%)", 10, 100, 40, help="Percentage of active trade profit to convert to Gold ETF.")
    
    # Fetch Data
    with st.spinner("Initializing Market Data & Calibrating Non-Linear Models..."):
        df = fetch_macro_data()
        current_gold_price = round(df['GOLDIETF.NS'].iloc[-1], 2)
        
        hurst_val = compute_hurst(df['GOLDIETF.NS'], window=hurst_window)
        lppls_res = calibrate_lppls(df['GOLDIETF.NS'].tail(lppls_window))
        macro_sentiment = fetch_sentiment_data()

    # --- UI Layout: Top Execution Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ICICI Gold ETF (GOLDIETF.NS)", f"₹{current_gold_price}")
    
    hurst_color = "normal" if hurst_val >= 0.65 else "inverse"
    col2.metric("Hurst Exponent (100d)", round(hurst_val, 3), delta="Persistent" if hurst_val >= 0.65 else "Mean-Reverting", delta_color=hurst_color)
    
    tc_val = lppls_res['tc_days_ahead']
    tc_color = "inverse" if lppls_res['exit_warning'] else "normal"
    col3.metric("LPPLS Criticality (Days to Crash)", f"{tc_val} days", delta="LIQUIDATION WARNING" if tc_val <= 30 else "Safe Acceleration", delta_color=tc_color)
    
    col4.metric("Macro NLP Sentiment (VADER)", macro_sentiment, delta="Bullish" if macro_sentiment > 0.05 else "Bearish" if macro_sentiment < -0.05 else "Neutral")

    # --- Containerized Tabs for Logical Separation ---
    tab1, tab2, tab3 = st.tabs(["Macro Correlation Matrix", "Power Law Criticality Chart", "Accumulation Sweep Calculator"])
    
    # TAB 1: Dynamic Correlation Matrix
    with tab1:
        st.subheader("Dynamic Macroeconomic Correlation (Rolling 60-Day Proxy)")
        st.markdown("Tracks shifting relationships between Gold, USD/INR, Crude, and Equities. Strong negative correlations against the Nifty 50 signify Gold transitioning into a Risk-Off safe-haven regime.")
        
        # Calculate rolling Pearson correlation as a lightweight proxy for full DCC-GARCH
        rolling_corr = df.pct_change().rolling(window=60).corr().dropna()
        latest_corr = rolling_corr.xs(rolling_corr.index.levels[0][-1])
        
        # Plotly Heatmap Rendering
        fig_corr = go.Figure(data=go.Heatmap(
            z=latest_corr.values,
            x=latest_corr.columns,
            y=latest_corr.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=np.round(latest_corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 14}
        ))
        fig_corr.update_layout(height=500, width=700)
        st.plotly_chart(fig_corr, use_container_width=True)

    # TAB 2: Price Action & Structural Moving Averages
    with tab2:
        st.subheader("Gold ETF Price Trajectory vs. Structural Averages")
        fig_px = go.Figure()
        fig_px.add_trace(go.Scatter(x=df.index, y=df['GOLDIETF.NS'], name='Gold ETF Close', line=dict(color='gold', width=2)))
        fig_px.add_trace(go.Scatter(x=df.index, y=df['GOLDIETF.NS'].rolling(20).mean(), name='20d SMA', line=dict(color='blue', dash='dot')))
        fig_px.add_trace(go.Scatter(x=df.index, y=df['GOLDIETF.NS'].rolling(200).mean(), name='200d SMA', line=dict(color='white')))
        fig_px.update_layout(template="plotly_dark", height=500, xaxis_title="Date", yaxis_title="Price (INR)")
        st.plotly_chart(fig_px, use_container_width=True)

    # TAB 3: Profit Sweep Calculator (Stateful)
    with tab3:
        st.subheader("Partial Profit Accumulation Engine")
        st.markdown(f"**Current Execution Directive:** Sweep **{sweep_pct}%** of realized short-term trading profits directly into unleveraged physical Gold ETF accumulation to convert active alpha into inflation-hedged beta.")
        
        # Form to prevent reruns until submission
        with st.form("sweep_form", clear_on_submit=True):
            trade_profit = st.number_input("Realized Profit from Closed Trade (INR)", min_value=0.0, step=1000.0)
            submitted = st.form_submit_button("Execute Profit Sweep")
            
            if submitted and trade_profit > 0:
                sweep_amt = trade_profit * (sweep_pct / 100.0)
                units_bought = math.floor(sweep_amt / current_gold_price)
                actual_cost = units_bought * current_gold_price
                
                # Append to Session State Ledger
                new_record = pd.DataFrame([{
                    'Date': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Trade_Profit_INR': trade_profit,
                    'Sweep_Amount_INR': actual_cost,
                    'Gold_ETF_Price': current_gold_price,
                    'Units_Accumulated': units_bought
                }])
                st.session_state.portfolio_sweeps = pd.concat([st.session_state.portfolio_sweeps, new_record], ignore_index=True)
                st.session_state.total_gold_units += units_bought
                st.success(f"Execution complete: Successfully swept ₹{actual_cost:,.2f} to acquire {units_bought} units of GOLDIETF.NS.")

        if not st.session_state.portfolio_sweeps.empty:
            st.write("### Tactical Accumulation Ledger")
            st.dataframe(st.session_state.portfolio_sweeps, use_container_width=True)
            total_value = st.session_state.total_gold_units * current_gold_price
            st.info(f"**Total Gold Vault Balance:** {st.session_state.total_gold_units} Units | **Current Market Value:** ₹{total_value:,.2f}")

if __name__ == "__main__":
    main()