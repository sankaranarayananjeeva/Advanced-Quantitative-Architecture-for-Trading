import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import re
import math
import time
from datetime import datetime, timedelta
import yfinance as yf
from scipy.optimize import minimize
from playwright.sync_api import sync_playwright

# Streamlit Cloud Playwright Installation Hook
os.system("playwright install chromium")

# ==========================================
# 1. Page Configuration & Styling
# ==========================================
st.set_page_config(page_title="Power Law Macro Quant", layout="wide")

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

st.title("Asymmetric Power Law & Macro Correlation Engine")
st.markdown("Integrates LPPLS Criticality, Multi-Source Fundamentals (Screener + Yahoo), Macro DCC-GARCH Signals, and Gold ETF Wealth Sweeping.")

# ==========================================
# 2. Sidebar Configuration
# ==========================================
st.sidebar.header("Strategy Configurations")

with st.sidebar.expander("Universe Selection", expanded=True):
    default_tickers = ["PFC", "RECLTD", "ZOMATO", "TATASTEEL", "ITC", "WIPRO", "ONGC"]
    selected_tickers = st.multiselect("Select NSE Equities (Without .NS)", options=default_tickers, default=default_tickers)

with st.sidebar.expander("Composite Signal Weightings"):
    w_fund = st.sidebar.slider("Fundamental Anchor (%)", 0, 100, 40)
    w_tech = st.sidebar.slider("Technical Validation (MRS) (%)", 0, 100, 20)
    w_plcs = st.sidebar.slider("Power Law Criticality (%)", 0, 100, 40)
    if (w_fund + w_tech + w_plcs) != 100:
        st.sidebar.error("Weights must sum exactly to 100%.")

with st.sidebar.expander("Gold ETF Sweep Rules"):
    profit_sweep_pct = st.slider("Net Profit Swept to Gold ETF (%)", 10, 100, 50)
    initial_capital = st.number_input("Initial Portfolio Capital (₹)", value=1000000, step=100000)

# ==========================================
# 3. Multi-Source Fundamental Extraction
# ==========================================
def safe_extract(source, key, default=None):
    try:
        val = source.get(key)
        return float(val) if val is not None else default
    except (AttributeError, TypeError, ValueError):
        return default

def extract_screener_playwright(symbol, page):
    """Scrapes Screener.in for accurate Indian fundamental data and Promoter Pledge."""
    fallback_data = {"promoter_pledge_pct": 0.0}
    try:
        response = page.goto(f"https://www.screener.in/company/{symbol}/consolidated/", timeout=15000, wait_until="domcontentloaded")
        if response and response.status == 404:
            page.goto(f"https://www.screener.in/company/{symbol}/", timeout=15000, wait_until="domcontentloaded")
            
        page.wait_for_timeout(1000)
        html = page.content()
        
        def extract_metric(name, html_text):
            pattern = fr'<span class="name">\s*{name}\s*</span>.*?<span class="number">([^<]+)</span>'
            match = re.search(pattern, html_text, re.DOTALL | re.IGNORECASE)
            if match:
                try: return float(match.group(1).replace(",", "").strip())
                except ValueError: return None
            return None

        fallback_data["pe_ratio"] = extract_metric(r"Stock P/E", html)
        fallback_data["roce_pct"] = extract_metric(r"ROCE", html)
        fallback_data["ev_ebitda"] = extract_metric(r"EV / EBITDA", html)
        
        current_price = extract_metric(r"Current Price", html)
        book_value = extract_metric(r"Book Value", html)
        if current_price and book_value and book_value != 0:
            fallback_data["pb_ratio"] = round(current_price / book_value, 2)

        match_top = re.search(r'Pledged percentage.*?<span class="number">([^<]+)</span>', html, re.DOTALL | re.IGNORECASE)
        match_any = re.search(r'Pledge[^\d]*?(\d+\.?\d*)%', html, re.IGNORECASE)
        
        if match_top:
            fallback_data["promoter_pledge_pct"] = float(match_top.group(1).replace(",", "").strip())
        elif match_any:
            fallback_data["promoter_pledge_pct"] = float(match_any.group(1).strip())

    except Exception:
        pass
    return fallback_data

def calculate_piotroski(ticker_obj):
    try:
        bs = ticker_obj.balance_sheet
        ic = ticker_obj.income_stmt
        cf = ticker_obj.cashflow
        if bs.shape[1] < 2 or ic.shape[1] < 2 or cf.shape[1] < 2: return 0 
        cy, py = bs.columns[0], bs.columns[1]
        score = 0
        
        def safe_get(df, row, col):
            return df.loc[row, col] if (row in df.index and not pd.isna(df.loc[row, col])) else 0

        ni_cy = safe_get(ic, 'Net Income', cy)
        ta_cy = safe_get(bs, 'Total Assets', cy)
        roa_cy = (ni_cy / ta_cy) if ta_cy else 0
        if roa_cy > 0: score += 1
            
        cfo_cy = safe_get(cf, 'Operating Cash Flow', cy)
        if cfo_cy > 0: score += 1
        if ta_cy and (cfo_cy / ta_cy) > roa_cy: score += 1
            
        ni_py = safe_get(ic, 'Net Income', py)
        ta_py = safe_get(bs, 'Total Assets', py)
        roa_py = (ni_py / ta_py) if ta_py else 0
        if roa_cy > roa_py: score += 1
            
        debt_cy, debt_py = safe_get(bs, 'Long Term Debt', cy), safe_get(bs, 'Long Term Debt', py)
        if ta_cy and ta_py and (debt_cy / ta_cy) < (debt_py / ta_py): score += 1
            
        ca_cy, cl_cy = safe_get(bs, 'Current Assets', cy), safe_get(bs, 'Current Liabilities', cy)
        ca_py, cl_py = safe_get(bs, 'Current Assets', py), safe_get(bs, 'Current Liabilities', py)
        if cl_cy and cl_py and (ca_cy / cl_cy) > (ca_py / cl_py): score += 1
            
        shares_cy, shares_py = safe_get(bs, 'Ordinary Shares Number', cy), safe_get(bs, 'Ordinary Shares Number', py)
        if shares_cy <= shares_py: score += 1
            
        gp_cy, rev_cy = safe_get(ic, 'Gross Profit', cy), safe_get(ic, 'Total Revenue', cy)
        gp_py, rev_py = safe_get(ic, 'Gross Profit', py), safe_get(ic, 'Total Revenue', py)
        if rev_cy and rev_py:
            if (gp_cy / rev_cy) > (gp_py / rev_py): score += 1
            if ta_cy and ta_py and (rev_cy / ta_cy) > (rev_py / ta_py): score += 1
            
        return score
    except Exception:
        return 0

@st.cache_data(ttl=3600)
def fetch_multi_source_data(symbols):
    records = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        page = context.new_page()

        for sym in symbols:
            yf_sym = f"{sym}.NS"
            ticker = yf.Ticker(yf_sym)
            info = ticker.info
            
            # Yahoo Extract
            pe_yf = safe_extract(info, "trailingPE")
            pb_yf = safe_extract(info, "priceToBook")
            ev_ebitda_yf = safe_extract(info, "enterpriseToEbitda")
            f_score = calculate_piotroski(ticker)
            
            # Screener.in Extract
            screener_data = extract_screener_playwright(sym, page)
            
            # Priority Logic: Screener.in (Local Indian context) > Yahoo Finance
            pe_ratio = screener_data.get("pe_ratio") or pe_yf
            pb_ratio = screener_data.get("pb_ratio") or pb_yf
            ev_ebitda = screener_data.get("ev_ebitda") or ev_ebitda_yf
            roce_pct = screener_data.get("roce_pct", 0.0)
            pledge_pct = screener_data.get("promoter_pledge_pct", 0.0)
            
            records.append({
                "Ticker": f"{sym}.NS",
                "PE Ratio": pe_ratio,
                "PB Ratio": pb_ratio,
                "EV/EBITDA": ev_ebitda,
                "ROCE (%)": roce_pct,
                "F-Score": f_score,
                "Pledge (%)": pledge_pct
            })
        browser.close()
    return pd.DataFrame(records)

# ==========================================
# 4. Power Law Math (Hurst & LPPLS)
# ==========================================
def compute_hurst(prices, window=100):
    prices = prices.dropna()
    if len(prices) < 40: return np.nan
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
            mean_adj = chunk - np.mean(chunk)
            cum_dev = np.cumsum(mean_adj)
            r_n = np.max(cum_dev) - np.min(cum_dev)
            s_n = np.std(chunk, ddof=1)
            if s_n > 1e-12 and r_n > 0:
                rs_values.append(r_n / s_n)
        if rs_values:
            log_n.append(math.log(n))
            log_rs.append(math.log(np.mean(rs_values)))
            
    if len(log_n) < 3: return np.nan
    slope, _ = np.polyfit(log_n, log_rs, 1)
    return float(np.clip(slope, 0.0, 1.0))

def calibrate_lppls(prices, window=120):
    clean_prices = prices.dropna()
    if len(clean_prices) < 50: return 999.0
    series = clean_prices.iloc[-min(window, len(clean_prices)):]
    y = np.log(series.values)
    t = np.arange(len(y), dtype=float)
    t_last = float(len(y) - 1)

    def solve_linear(tc, m, omega):
        dt_vec = tc - t
        if np.any(dt_vec <= 0.5): return None
        f = np.power(dt_vec, m)
        g = f * np.cos(omega * np.log(dt_vec))
        h = f * np.sin(omega * np.log(dt_vec))
        X = np.column_stack([np.ones(len(t)), f, g, h])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            sse = np.sum((y - (X @ beta)) ** 2)
            return beta, sse
        except: return None

    best_sse, best_tc = float("inf"), t_last + 60.0
    for tc_cand in [t_last + 15, t_last + 30, t_last + 60, t_last + 120]:
        res = solve_linear(tc_cand, 0.5, 8.0)
        if res and res[1] < best_sse:
            best_sse, best_tc = res[1], tc_cand
            
    return float(best_tc - t_last)

def compute_zscore(series, invert=False):
    vals = series.astype(float).fillna(0).copy()
    if invert: vals = -1.0 * vals
    mean_val, std_val = vals.mean(), vals.std()
    if std_val < 1e-12 or pd.isna(std_val): return pd.Series(0.0, index=series.index)
    return (vals - mean_val) / std_val

# ==========================================
# 5. Execution & Rendering
# ==========================================
if st.button("Execute Multi-Source Quant Analysis"):
    with st.spinner("Extracting Macro Data & Cross-Asset Correlations..."):
        macro_tickers = {"DXY": "DX-Y.NYB", "Crude": "BZ=F", "Nifty": "^NSEI", "Gold": "ICICIGOLD.NS"}
        macro_df = yf.download(list(macro_tickers.values()), period="200d", progress=False)['Close'].ffill().dropna()
        
    st.header("1. Macroeconomic Environment & Dynamic Correlation")
    if not macro_df.empty:
        corr_matrix = macro_df.tail(60).pct_change().corr()
        col1, col2 = st.columns([1.2, 2])
        
        with col1:
            dxy_trend = macro_df['DX-Y.NYB'].iloc[-1] - macro_df['DX-Y.NYB'].iloc[-20]
            crude_trend = macro_df['BZ=F'].iloc[-1] - macro_df['BZ=F'].iloc[-20]
            nifty_trend = macro_df['^NSEI'].iloc[-1] - macro_df['^NSEI'].iloc[-20]
            
            if dxy_trend > 0 and crude_trend > 0 and nifty_trend < 0:
                macro_signal, macro_multiplier, css_class = "STRONG RISK-OFF", 0.50, "macro-off"
            elif dxy_trend < 0 and crude_trend < 0 and nifty_trend > 0:
                macro_signal, macro_multiplier, css_class = "STRONG RISK-ON", 1.25, "macro-on"
            else:
                macro_signal, macro_multiplier, css_class = "NEUTRAL", 1.00, "macro-neutral"
                
            st.markdown(f"""
            <div data-testid="metric-container" class="{css_class}">
                <p style="font-size:14px; color:#888; margin-bottom:0;">Macro Regime Classification</p>
                <h2 style="margin:0; padding:0;">{macro_signal}</h2>
                <p style="color:#aaa; font-size:12px; margin-top:5px;">PLCS Weight Multiplier: {macro_multiplier}x</p>
            </div><br>
            """, unsafe_allow_html=True)
            
        with col2:
            st.write("**Cross-Asset Conditional Correlation Matrix (Trailing 60 Days)**")
            fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
            fig_corr.update_layout(margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_corr, use_container_width=True)

    st.header("2. Power Law Criticality Ranker (Live Processing)")
    with st.spinner("Scraping Fundamentals (Screener + Yahoo) & Calibrating LPPLS Models..."):
        fund_df = fetch_multi_source_data(selected_tickers)
        price_df = yf.download([f"{s}.NS" for s in selected_tickers], period="200d", progress=False)['Close']
        
        fund_df["Hurst (100d)"] = fund_df["Ticker"].apply(lambda t: compute_hurst(price_df[t]) if t in price_df else np.nan)
        fund_df["LPPLS t_c (days)"] = fund_df["Ticker"].apply(lambda t: calibrate_lppls(price_df[t]) if t in price_df else np.nan)
        
        # Z-Score Aggregation
        fund_df["Z_Fund"] = compute_zscore(fund_df["ROCE (%)"]) + compute_zscore(fund_df["F-Score"]) + compute_zscore(fund_df["EV/EBITDA"], invert=True)
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

    st.header("3. Systematic Wealth Accumulation: ICICI Gold ETF Sweep Protocol")
    col_g1, col_g2 = st.columns([1, 1.5])
    with col_g1:
        sim_years = st.number_input("Simulation Horizon (Years)", min_value=1, max_value=20, value=5)
        est_eq_ret = 0.24
        est_gold_ret = 0.11
        st.info(f"**Calculated Sweep Ratio:** {profit_sweep_pct}% of net post-tax equity profit is diverted to Gold ETF.")

    with col_g2:
        eq_cap, gold_cap = [float(initial_capital)], [0.0]
        total_swept = 0.0
        
        for y in range(1, int(sim_years) + 1):
            gross_eq_prof = eq_cap[-1] * est_eq_ret
            net_eq_prof = gross_eq_prof * 0.80 # 20% STCG
            sweep = net_eq_prof * (profit_sweep_pct / 100)
            total_swept += sweep
            
            eq_cap.append(eq_cap[-1] + (net_eq_prof - sweep))
            gold_cap.append(gold_cap[-1] * (1 + est_gold_ret) + sweep)

        final_gold_val = gold_cap[-1]
        gold_ltcg = max(0.0, final_gold_val - total_swept) * 0.125
        term_net = eq_cap[-1] + (final_gold_val - gold_ltcg)

        st.write(f"### Terminal Metrics (Year {int(sim_years)})")
        st.write(f"- **Final Equity Portfolio Value:** ₹ {eq_cap[-1]:,.2f}")
        st.write(f"- **Gross Gold ETF Value:** ₹ {final_gold_val:,.2f}")
        st.success(f"**Total Net Wealth (Post All Taxes):** ₹ {term_net:,.2f}")

    results_df = pd.DataFrame({"Year": np.arange(0, sim_years + 1), "Equity Base (₹)": eq_cap, "Gold ETF Base (₹)": gold_cap})
    fig_growth = px.bar(results_df, x="Year", y=["Equity Base (₹)", "Gold ETF Base (₹)"], title="Portfolio Compounding Velocity", color_discrete_sequence=["#1f77b4", "#ffd700"])
    fig_growth.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_growth, use_container_width=True)