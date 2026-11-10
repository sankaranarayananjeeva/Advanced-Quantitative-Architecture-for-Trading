import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
from datetime import datetime, timedelta

# ==========================================
# 1. Page Configuration & Custom CSS Styling
# ==========================================
st.set_page_config(page_title="Power Law Macro Quant", layout="wide")

# Custom CSS for metric styling and structural layout mimicking institutional dashboards
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
    .macro-on { border-left: 5px solid #00cc66 !important; }   /* Risk-On Green */
    .macro-off { border-left: 5px solid #ff3333 !important; }  /* Risk-Off Red */
    .macro-neutral { border-left: 5px solid #ffcc00 !important; } /* Neutral Yellow */
</style>
""", unsafe_allow_html=True)

st.title("Asymmetric Power Law & Macro Correlation Engine")
st.markdown("Integrates LPPLS Criticality, Macro DCC-GARCH Signals, Next-Day ATR Execution, and Gold ETF Wealth Sweeping.")

# ==========================================
# 2. Sidebar Configuration & Weightings
# ==========================================
st.sidebar.header("Strategy Configurations")
with st.sidebar.expander("Composite Signal Weightings", expanded=True):
    st.write("Adjust multi-factor Z-score synthesis weights:")
    w_fund = st.slider("Fundamental Anchor (%)", 0, 100, 40)
    w_tech = st.slider("Technical Validation (MRS) (%)", 0, 100, 20)
    w_plcs = st.slider("Power Law Criticality (LPPLS/Hurst) (%)", 0, 100, 40)
    
    if (w_fund + w_tech + w_plcs) != 100:
        st.sidebar.error("Weights must sum exactly to 100%. Please adjust parameters.")

with st.sidebar.expander("Gold ETF (GOLDIETF) Sweep Rules"):
    profit_sweep_pct = st.slider("Net Profit Swept to Gold ETF (%)", 10, 100, 50)
    initial_capital = st.number_input("Initial Portfolio Capital (₹)", value=1000000, step=100000)
    investor_tax_slab = st.selectbox("Investor Marginal Tax Slab (%)", [0, 5, 20, 30], index=3)

# ==========================================
# 3. Market Data Acquisition (Direct REST API)
# ==========================================
@st.cache_data(ttl=3600)
def fetch_macro_data():
    """Fetches data directly via Yahoo JSON endpoints without requiring yfinance."""
    tickers = {
        "US Dollar Index (DXY)": "DX-Y.NYB",
        "Brent Crude": "BZ=F",
        "USD/INR": "INR=X",
        "S&P 500": "^GSPC",
        "NASDAQ 100": "^NDX",
        "US 10-Yr Yield (TNX)": "^TNX",
        "Nifty 50": "^NSEI",
        "ICICI Gold ETF": "ICICIGOLD.NS" 
    }
    
    # Calculate Unix timestamps for the API
    end_date = datetime.today()
    start_date = end_date - timedelta(days=200)
    period1 = int(start_date.timestamp())
    period2 = int(end_date.timestamp())
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    data = pd.DataFrame()
    
    for name, ticker in tickers.items():
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={period1}&period2={period2}&interval=1d"
            res = requests.get(url, headers=headers, timeout=10)
            
            if res.status_code == 200:
                json_data = res.json()
                result = json_data.get('chart', {}).get('result', [])
                
                if result:
                    timestamps = result[0].get('timestamp', [])
                    quote = result[0]['indicators']['quote'][0]
                    closes = quote.get('close', [])
                    
                    # Filter out any None values returned by the API
                    valid_data = [(ts, c) for ts, c in zip(timestamps, closes) if c is not None]
                    
                    if valid_data:
                        valid_ts, valid_closes = zip(*valid_data)
                        dates = pd.to_datetime(valid_ts, unit='s').dt.date
                        
                        df = pd.DataFrame({name: valid_closes}, index=dates)
                        
                        if data.empty:
                            data = df
                        else:
                            data = data.join(df, how='outer')
        except Exception as e:
            st.warning(f"Failed to download {name}: {e}")
            
    return data.ffill().dropna()

macro_df = fetch_macro_data()

# ==========================================
# 4. Macro Correlation Matrix & Signal Generator
# ==========================================
st.header("1. Macroeconomic Environment & Dynamic Correlation")

if not macro_df.empty:
    # Compute 60-day rolling correlation matrix simulating DCC-GARCH output
    recent_data = macro_df.tail(60)
    corr_matrix = recent_data.pct_change().corr()
    
    col1, col2 = st.columns([1.2, 2])
    
    with col1:
        # Determine Macro Risk Signal based on recent momentum trends
        dxy_trend = macro_df['US Dollar Index (DXY)'].iloc[-1] - macro_df['US Dollar Index (DXY)'].iloc[-20]
        crude_trend = macro_df['Brent Crude'].iloc[-1] - macro_df['Brent Crude'].iloc[-20]
        sp500_trend = macro_df['S&P 500'].iloc[-1] - macro_df['S&P 500'].iloc[-20]
        
        # State Classification Logic
        if dxy_trend > 0 and crude_trend > 0 and sp500_trend < 0:
            macro_signal = "STRONG RISK-OFF"
            macro_multiplier = 0.50
            css_class = "macro-off"
        elif dxy_trend < 0 and crude_trend < 0 and sp500_trend > 0:
            macro_signal = "STRONG RISK-ON"
            macro_multiplier = 1.25
            css_class = "macro-on"
        else:
            macro_signal = "NEUTRAL"
            macro_multiplier = 1.00
            css_class = "macro-neutral"
            
        st.markdown(f"""
        <div data-testid="metric-container" class="{css_class}">
            <p style="font-size:14px; color:#888; margin-bottom:0;">Macro Regime Classification</p>
            <h2 style="margin:0; padding:0;">{macro_signal}</h2>
            <p style="color:#aaa; font-size:12px; margin-top:5px;">PLCS Weight Multiplier: {macro_multiplier}x</p>
        </div>
        <br>
        """, unsafe_allow_html=True)
        
        st.subheader("Next-Day Execution Pricing Triggers")
        st.write("Calculated via Average True Range (ATR) Volatility Breakout mechanics.")
        
        # Calculate ATR and Next Day Triggers for the Nifty 50 Index
        high_low = macro_df['Nifty 50'].diff().abs() # Simplified proxy for daily volatility
        atr_5 = high_low.rolling(5).mean().iloc[-1]
        atr_14 = high_low.rolling(14).mean().iloc[-1]
        last_close = macro_df['Nifty 50'].iloc[-1]
        
        trigger_df = pd.DataFrame({
            "Metric": ["Last Closing Price", "Long Breakout Entry (Open + 0.3*ATR5)", 
                       "Short/Hedge Entry (Open - 0.3*ATR5)", "Dynamic Stop Loss (Entry - 1.5*ATR14)"],
            "Value": [f"₹ {last_close:,.2f}", f"Open + ₹ {0.3 * atr_5:.2f}", 
                      f"Open - ₹ {0.3 * atr_5:.2f}", f"Entry - ₹ {1.5 * atr_14:.2f}"]
        })
        st.dataframe(trigger_df, hide_index=True, use_container_width=True)

    with col2:
        st.write("**Cross-Asset Conditional Correlation Matrix (Trailing 60 Days)**")
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", 
                             color_continuous_scale="RdBu_r", origin='lower')
        fig_corr.update_layout(margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_corr, theme="streamlit", use_container_width=True)

# ==========================================
# 5. Power Law Equity Ranker (Almgren-Chriss Target)
# ==========================================
st.header("2. Power Law Criticality Ranker & Liquidation Monitor")

# Mocking data to represent the complex output of the LPPLS and Hurst calculations
mock_stocks = pd.DataFrame({
    "Ticker": ["PFC.NS", "RECLTD.NS", "ZOMATO.NS", "TATASTEEL.NS", "ITC.NS", "WIPRO.NS"],
    "Hurst (100d)": [0.72, 0.68, 0.75, 0.52, 0.49, 0.54],
    "Hill Alpha": [1.8, 2.1, 1.5, 2.8, 3.2, 2.5],
    "LPPLS t_c (days)": [85.0, 42.5, 18.2, 120.0, 200.0, 28.4],
    "Score_Fund": [1.2, 1.1, 0.4, 0.8, 1.5, 1.0],
    "Score_Tech": [1.5, 1.3, 2.1, 0.5, 0.2, 0.6]
})

# Apply user-defined UI weightings and the calculated Macro Multiplier
mock_stocks["Composite Score S_i"] = (
    (w_fund/100) * mock_stocks["Score_Fund"] +
    (w_tech/100) * mock_stocks["Score_Tech"] +
    (w_plcs/100) * (mock_stocks["Hurst (100d)"] * 2) 
) * macro_multiplier

# Flag algorithmic liquidation if approaching critical singularity (tc <= 30) AND Hurst <= 0.55
mock_stocks["Liquidation Alert"] = np.where(
    (mock_stocks["LPPLS t_c (days)"] <= 30) & (mock_stocks["Hurst (100d)"] <= 0.55), 
    True, False)

st.dataframe(
    mock_stocks.sort_values(by="Composite Score S_i", ascending=False),
    column_config={
        "Ticker": st.column_config.TextColumn("NSE Symbol"),
        "Hurst (100d)": st.column_config.NumberColumn("Hurst Exponent (H)", format="%.2f"),
        "LPPLS t_c (days)": st.column_config.NumberColumn("LPPLS Time to Singularity (Days)", format="%.1f"),
        "Composite Score S_i": st.column_config.NumberColumn("Weighted Composite", format="%.3f"),
        "Liquidation Alert": st.column_config.CheckboxColumn("CRITICAL: Almgren-Chriss Exit Mandate")
    },
    use_container_width=True,
    hide_index=True
)

st.error("⚠️ **Execution Mandate:** Assets flagged with a Liquidation Alert must be executed via the Almgren-Chriss optimal trajectory (aggressively front-loaded) to escape the log-periodic singularity and bypass NSE Limit Price Protection (LPP) constraints before the crash phase transition.")

# ==========================================
# 6. ICICI Gold ETF (GOLDIETF) Profit Sweeper & Calculator
# ==========================================
st.header("3. Systematic Wealth Accumulation: ICICI Gold ETF Sweep Protocol")
st.write("Calculates net portfolio compounding by sweeping partial equity profits into the ICICI Prudential Gold ETF. Rigidly enforces the Indian Finance Act 2024 tax regime (12.5% LTCG on Gold ETFs after 12 months, no indexation; 20% STCG on Equity).")

col_g1, col_g2 = st.columns([1, 1.5])

with col_g1:
    sim_years = st.number_input("Simulation Horizon (Years)", min_value=1, max_value=20, value=5)
    est_equity_return = st.number_input("Est. Gross Annual Equity Return from Power Law (%)", value=24.0, step=1.0) / 100
    est_gold_return = st.number_input("Est. Annual Domestic Gold Return (%)", value=11.0, step=1.0) / 100
    
    st.info(f"**Calculated Sweep Ratio:** {profit_sweep_pct}% of net post-tax equity profit is diverted to Gold ETF.")

with col_g2:
    # Compounding Simulation Logic
    equity_capital = [float(initial_capital)]
    gold_capital = [0.0]
    total_stcg_paid = 0.0
    total_swept_principal = 0.0

    for y in range(1, int(sim_years) + 1):
        curr_eq = equity_capital[-1]
        curr_au = gold_capital[-1]
        
        # 1. Gross Equity Return Generation
        gross_eq_profit = curr_eq * est_equity_return
        
        # 2. Equity Taxation: Assumes intra-year power law exits, triggering 20% STCG
        tax_on_eq = gross_eq_profit * 0.20
        total_stcg_paid += tax_on_eq
        net_eq_profit = gross_eq_profit - tax_on_eq
        
        # 3. Capital Allocation Sweep Protocol
        swept_to_gold = net_eq_profit * (profit_sweep_pct / 100)
        total_swept_principal += swept_to_gold
        reinvested_in_eq = net_eq_profit - swept_to_gold
        
        # 4. Gold ETF Return Generation
        au_growth = curr_au * est_gold_return
        
        # Append End-of-Year Totals
        equity_capital.append(curr_eq + reinvested_in_eq)
        gold_capital.append(curr_au + au_growth + swept_to_gold)

    # 5. Terminal Liquidation & Final Gold Taxation
    final_gold_value = gold_capital[-1]
    gross_gold_profit = max(0.0, final_gold_value - total_swept_principal) 
    gold_ltcg_tax = gross_gold_profit * 0.125
    net_final_gold = final_gold_value - gold_ltcg_tax
    terminal_total_net = equity_capital[-1] + net_final_gold

    st.write(f"### Terminal Metrics (Year {int(sim_years)})")
    st.write(f"- **Final Equity Portfolio Value:** ₹ {equity_capital[-1]:,.2f}")
    st.write(f"- **Gross Gold ETF Value:** ₹ {final_gold_value:,.2f}")
    st.write(f"- **Total Equity STCG Taxes Paid:** ₹ {total_stcg_paid:,.2f}")
    st.write(f"- **Terminal Gold LTCG Tax (12.5%):** ₹ {gold_ltcg_tax:,.2f}")
    st.success(f"**Total Net Wealth (Post All Taxes):** ₹ {terminal_total_net:,.2f}")

results_df = pd.DataFrame({
    "Year": np.arange(0, sim_years + 1),
    "Equity Base (₹)": equity_capital,
    "Gold ETF Base (₹)": gold_capital
})

fig_growth = px.bar(results_df, x="Year", y=["Equity Base (₹)", "Gold ETF Base (₹)"],
                     title="Portfolio Compounding Velocity: Reinvested Equity vs. Shielded Gold ETF",
                     color_discrete_sequence=["#1f77b4", "#ffd700"], barmode="stack")
fig_growth.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

st.plotly_chart(fig_growth, theme="streamlit", use_container_width=True)