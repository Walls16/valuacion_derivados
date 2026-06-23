"""
Page 7 — Subyacentes en Vivo
Real-time option pricing via yfinance: spot price, option chain, IV extraction, ATM pricing.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, date
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import BSMEngine, HestonEngine, MertonEngine, compare_all_models

try:
    import yfinance as yf
    YFINANCE_OK = True
except ImportError:
    YFINANCE_OK = False

st.set_page_config(page_title="Live · VQD", page_icon="∂", layout="wide")

dark = st.session_state.get("dark_mode", True)
bg        = "#0d0f14" if dark else "#f4f6fb"
surface   = "#131720" if dark else "#ffffff"
card      = "#1a1f2e" if dark else "#ffffff"
border    = "#2a3040" if dark else "#dde3ef"
text_main = "#e8eaf0" if dark else "#1a2035"
text_sub  = "#8892a4" if dark else "#6b7a99"
accent    = "#4f8ef7" if dark else "#2563eb"
positive  = "#34d399"
negative  = "#f87171"
grid_col  = "#1e2535" if dark else "#e8edf5"
plot_bg   = "#131720" if dark else "#ffffff"
paper_bg  = "#0d0f14" if dark else "#f4f6fb"

MODEL_COLORS = {
    "BSM":    "#4f8ef7",
    "CRR":    "#34d399",
    "Heston": "#a78bfa",
    "Merton": "#f59e0b",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Playfair+Display:wght@700&display=swap');
html, body, [data-testid="stApp"] {{ background-color: {bg} !important; color: {text_main} !important; font-family: 'Inter', sans-serif; }}
[data-testid="stSidebar"] {{ background-color: {surface} !important; border-right: 1px solid {border} !important; }}
[data-testid="stSidebar"] * {{ color: {text_main} !important; }}
.main .block-container {{ padding-top: 1.5rem; max-width: 1200px; }}
.page-title {{ font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 700; color: {text_main}; }}
.page-eyebrow {{ font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.16em; color: {accent}; text-transform: uppercase; margin-bottom: 0.5rem; }}
.page-sub {{ font-size: 0.9rem; color: {text_sub}; margin-bottom: 1.5rem; }}
.ticker-card {{ background: {card}; border: 1px solid {border}; border-radius: 10px; padding: 1.2rem 1.5rem; display: flex; gap: 2rem; align-items: center; margin-bottom: 1rem; }}
.ticker-symbol {{ font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 700; color: {text_main}; }}
.ticker-price {{ font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 600; color: {accent}; }}
.ticker-change-pos {{ font-size: 0.95rem; color: {positive}; }}
.ticker-change-neg {{ font-size: 0.95rem; color: {negative}; }}
.ticker-meta {{ font-size: 0.78rem; color: {text_sub}; }}
.chain-header {{ font-size: 0.7rem; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.16em; text-transform: uppercase; color: {text_sub}; margin: 1.2rem 0 0.6rem 0; }}
.atm-badge {{ display: inline-block; background: #f59e0b22; color: #f59e0b; border: 1px solid #f59e0b44; border-radius: 4px; padding: 0.1rem 0.5rem; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; margin-left: 0.4rem; }}
.itm-badge {{ display: inline-block; background: {positive}22; color: {positive}; border: 1px solid {positive}44; border-radius: 4px; padding: 0.1rem 0.5rem; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; margin-left: 0.4rem; }}
.otm-badge {{ display: inline-block; background: {negative}22; color: {negative}; border: 1px solid {negative}44; border-radius: 4px; padding: 0.1rem 0.5rem; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; margin-left: 0.4rem; }}
[data-testid="metric-container"] {{ background: {card} !important; border: 1px solid {border} !important; border-radius: 8px !important; padding: 0.8rem 1rem !important; }}
[data-testid="metric-container"] label {{ color: {text_sub} !important; font-size: 0.75rem !important; }}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{ color: {text_main} !important; font-family: 'JetBrains Mono', monospace !important; }}
.stTabs [data-baseweb="tab"] {{ color: {text_sub} !important; }}
.stTabs [aria-selected="true"] {{ color: {accent} !important; border-bottom-color: {accent} !important; }}
label {{ color: {text_sub} !important; font-size: 0.82rem !important; }}
.dataframe {{ background: {card} !important; color: {text_main} !important; }}
thead tr th {{ background: {surface} !important; color: {text_sub} !important; font-size: 0.75rem !important; }}
tbody tr td {{ font-family: 'JetBrains Mono', monospace !important; font-size: 0.78rem !important; }}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ∂ VQD")
    st.caption("Valuación Cuantitativa de Derivados")
    st.divider()
    dark_toggle = st.toggle("Modo oscuro", value=dark, key="theme_toggle")
    if dark_toggle != dark:
        st.session_state.dark_mode = dark_toggle
        st.rerun()
    st.divider()

st.markdown('<div class="page-eyebrow">Datos en Tiempo Real · Página 7</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Subyacentes en Vivo</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Valuación de opciones sobre datos reales de mercado vía yfinance. Extrae precio spot, cadena de opciones e IV implícita del mercado.</div>', unsafe_allow_html=True)

if not YFINANCE_OK:
    st.error("yfinance no está instalado. Ejecuta: `pip install yfinance`")
    st.stop()

def plotly_layout(title=""):
    return dict(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor=paper_bg, plot_bgcolor=plot_bg,
        font=dict(family="Inter", color=text_main, size=12),
        title=dict(text=title, font=dict(size=14, color=text_main)),
        xaxis=dict(gridcolor=grid_col),
        yaxis=dict(gridcolor=grid_col),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=40, b=40),
    )

#  Ticker input 
col_ticker, col_r_input = st.columns([2, 1])
with col_ticker:
    ticker_input = st.text_input(
        "Símbolo del ticker (yfinance)",
        value="NKE",
        placeholder="NKE, AAPL, MSFT, AMZN, SPY...",
        help="Cualquier símbolo válido en Yahoo Finance"
    ).upper().strip()

with col_r_input:
    r_manual = st.number_input("Tasa libre de riesgo (r, %)", value=5.25, step=0.25, format="%.2f") / 100

st.divider()

#  Fetch data 
@st.cache_data(ttl=120, show_spinner=False)
def fetch_ticker_data(ticker: str):
    """Returns only pickle-serializable data — no Ticker object."""
    try:
        tkr  = yf.Ticker(ticker)
        info = dict(tkr.info)

        # yfinance >= 0.2.x can return MultiIndex columns — flatten them
        hist = tkr.history(period="5d", auto_adjust=True)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = [c[0] for c in hist.columns]
        hist = hist.reset_index()

        expirations = list(tkr.options)
        return info, hist, expirations, None
    except Exception as e:
        return None, None, None, str(e)

with st.spinner(f"Obteniendo datos de {ticker_input}..."):
    info, hist, expirations, err = fetch_ticker_data(ticker_input)

if err or info is None or hist is None or hist.empty:
    st.error(f"No se pudieron obtener datos para **{ticker_input}**. Verifica el simbolo.")
    st.stop()

#  Spot price — robust extraction 
def _safe_col(df, *names):
    """Return the first matching column (case-insensitive), or a Series of NaN."""
    cols_lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in cols_lower:
            return df[cols_lower[n.lower()]]
    return pd.Series([float("nan")] * len(df))

close_series  = _safe_col(hist, "Close", "close", "Adj Close")
volume_series = _safe_col(hist, "Volume", "volume")

# Drop NaN rows before reading prices
close_clean = close_series.dropna()
if close_clean.empty:
    st.error(f"No se pudieron obtener precios para **{ticker_input}**.")
    st.stop()

spot       = float(close_clean.iloc[-1])
prev_close = float(close_clean.iloc[-2]) if len(close_clean) >= 2 else spot
change     = spot - prev_close
change_pct = (change / prev_close * 100) if prev_close != 0 else 0.0
volume     = float(volume_series.dropna().iloc[-1]) if not volume_series.dropna().empty else 0

name      = info.get("longName", ticker_input) or ticker_input
currency  = info.get("currency", "USD") or "USD"
sector    = info.get("sector", "N/A") or "N/A"

# dividendYield: yfinance returns decimal (0.02 = 2%), but some tickers return 0 or None
_dy = info.get("dividendYield", None) or info.get("trailingAnnualDividendYield", None) or 0.0
# Guard against already-scaled values (> 1 means it came as percentage)
div_yield = float(_dy) / 100.0 if float(_dy) > 1.0 else float(_dy)

#  Ticker info card 
change_class = "ticker-change-pos" if change >= 0 else "ticker-change-neg"
change_arrow = "" if change >= 0 else ""
st.markdown(f"""
<div class="ticker-card">
    <div>
        <div class="ticker-symbol">{ticker_input}</div>
        <div class="ticker-meta">{name}</div>
        <div class="ticker-meta">{sector} · {currency}</div>
    </div>
    <div>
        <div class="ticker-price">{currency} {spot:.2f}</div>
        <div class="{change_class}">{change_arrow} {change:+.2f} ({change_pct:+.2f}%)</div>
        <div class="ticker-meta">Dividendo: {div_yield*100:.2f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)

#  Historical chart 
@st.cache_data(ttl=300)
def fetch_history(ticker: str, period: str):
    h = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if isinstance(h.columns, pd.MultiIndex):
        h.columns = [c[0] for c in h.columns]
    return h.reset_index()

hist_long = fetch_history(ticker_input, "6mo")
_close_long = _safe_col(hist_long, "Close", "close")
_open_long  = _safe_col(hist_long, "Open",  "open")
_date_long  = _safe_col(hist_long, "Date",  "Datetime", "index")
if not hist_long.empty and not _close_long.dropna().empty:
    fig_hist = go.Figure()
    colors_hist = [positive if c >= o else negative
                   for c, o in zip(_close_long.fillna(method="ffill"), _open_long.fillna(method="ffill"))]
    fig_hist.add_trace(go.Scatter(
        x=_date_long, y=_close_long,
        mode="lines", line=dict(color=accent, width=1.5),
        fill="tozeroy",
        fillcolor=f"rgba(79,142,247,0.07)" if dark else "rgba(37,99,235,0.07)",
        name="Precio cierre",
    ))
    fig_hist.update_layout(**plotly_layout(f"{ticker_input} — Precio histórico (6 meses)"), height=260)
    fig_hist.update_xaxes(title="")
    fig_hist.update_yaxes(title=f"Precio ({currency})")
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

#  Expiration selector 
if not expirations:
    st.warning("No hay opciones disponibles para este ticker.")
    st.stop()

st.markdown('<div class="chain-header">Cadena de opciones</div>', unsafe_allow_html=True)

col_exp, col_opt_type = st.columns([2, 1])
with col_exp:
    exp_choice = st.selectbox("Fecha de vencimiento", expirations[:12])
with col_opt_type:
    opt_type_live = st.radio("Tipo", ["call", "put"], horizontal=True)

#  Fetch option chain 
@st.cache_data(ttl=120)
def fetch_chain(ticker: str, expiry: str):
    try:
        chain = yf.Ticker(ticker).option_chain(expiry)
        def _flatten(df):
            """Normalize column names to lowercase and flatten MultiIndex if needed."""
            if isinstance(df.columns, pd.MultiIndex):
                df = df.copy()
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df = df.copy()
            df.columns = [str(c).strip() for c in df.columns]
            return df.reset_index(drop=True)
        return _flatten(chain.calls), _flatten(chain.puts), None
    except Exception as e:
        return None, None, str(e)

with st.spinner("Cargando cadena de opciones..."):
    calls_df, puts_df, chain_err = fetch_chain(ticker_input, exp_choice)

if chain_err or calls_df is None or calls_df.empty:
    st.error(f"Error cargando cadena de opciones para {exp_choice}. Prueba otra fecha de vencimiento.")
    st.stop()

# Normalize column names to lowercase for robust access
def _col(df, *names):
    """Get first matching column by case-insensitive name."""
    cols_lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in cols_lower:
            return cols_lower[n.lower()]
    return None

#  Time to expiry 
exp_date = datetime.strptime(exp_choice, "%Y-%m-%d").date()
T_live = max((exp_date - date.today()).days / 365.0, 1 / 365)
q_live = div_yield

_raw = calls_df if opt_type_live == "call" else puts_df
df_chain = _raw.copy()

# Standardize column names to lowercase for consistent access
df_chain.columns = [c.lower() for c in df_chain.columns]

# Filter & clean: widen range progressively until we have at least one strike
df_chain = df_chain[df_chain["strike"] > 0].copy()
for _pct in [0.30, 0.50, 1.00]:
    _lo = spot * (1 - _pct)
    _hi = spot * (1 + _pct)
    _f  = df_chain[(df_chain["strike"] >= _lo) & (df_chain["strike"] <= _hi)]
    if not _f.empty:
        df_chain = _f.copy()
        break

if df_chain.empty:
    st.error(f"La cadena de opciones de {ticker_input} / {exp_choice} no tiene strikes disponibles. Prueba otra fecha.")
    st.stop()

# Always reset index after filtering so loc[] never gets stale integer keys
df_chain = df_chain.reset_index(drop=True)

df_chain["midPrice"] = (df_chain["bid"] + df_chain["ask"]) / 2
df_chain["midPrice"] = df_chain["midPrice"].clip(lower=0.001)

# IV extraction
bsm_iv_base = BSMEngine(spot, spot, T_live, r_manual, 0.3, q_live)

def safe_iv(row):
    try:
        mid = row["midPrice"]
        if mid <= 0.001:
            return np.nan
        eng = BSMEngine(spot, row["strike"], T_live, r_manual, 0.3, q_live)
        iv = eng.implied_vol(mid, opt_type_live)
        return iv * 100 if (not np.isnan(iv) and 1 < iv * 100 < 250) else np.nan
    except:
        return np.nan

df_chain["IV_market"] = df_chain.apply(safe_iv, axis=1)

# ATM strike — use integer position (iloc) so NaN index is impossible
_atm_pos        = int((df_chain["strike"] - spot).abs().values.argmin())
atm_idx_auto    = df_chain.index[_atm_pos]
atm_strike_auto = df_chain.loc[atm_idx_auto, "strike"]
_iv_auto_raw    = df_chain.loc[atm_idx_auto, "IV_market"]
atm_iv_auto     = _iv_auto_raw if (not isinstance(_iv_auto_raw, float) or not np.isnan(_iv_auto_raw)) else None

#  Strike selector 
st.markdown(f"<div class='chain-header'>Valuación de opción — {ticker_input}</div>", unsafe_allow_html=True)

col_strike_mode, col_strike_val, col_sigma_src = st.columns([1, 1.2, 1.2])

with col_strike_mode:
    strike_mode = st.radio(
        "Strike",
        ["ATM automático", "Manual"],
        help="ATM automático usa el strike más cercano al spot. Manual permite cualquier valor.",
    )

with col_strike_val:
    if strike_mode == "Manual":
        user_strike = st.number_input(
            "Strike personalizado (K)",
            min_value=float(max(spot * 0.3, 0.01)),
            max_value=float(spot * 3.0),
            value=float(round(spot, 2)),
            step=0.5,
            format="%.2f",
        )
        chosen_strike = user_strike
        # closest chain row for market reference
        _diffs  = (df_chain["strike"] - user_strike).abs()
        _ref_pos = int(_diffs.values.argmin()) if not _diffs.empty else 0
        ref_idx  = df_chain.index[_ref_pos]
    else:
        chosen_strike = atm_strike_auto
        ref_idx = atm_idx_auto
        st.markdown(
            f"<div style='padding-top:1.7rem;font-family:JetBrains Mono,monospace;"
            f"font-size:1rem;color:{accent}'>K = ${chosen_strike:.2f}</div>",
            unsafe_allow_html=True,
        )

with col_sigma_src:
    ref_iv_raw = df_chain.loc[ref_idx, "IV_market"] if ref_idx in df_chain.index else np.nan
    ref_strike_label = df_chain.loc[ref_idx, "strike"] if ref_idx in df_chain.index else "---"

    # Sanitize: yfinance can return absurd IVs on illiquid strikes -> clip to [1%, 250%]
    _IV_MIN, _IV_MAX = 0.01, 2.50
    if (not np.isnan(ref_iv_raw)) and (_IV_MIN <= ref_iv_raw <= _IV_MAX):
        ref_iv = ref_iv_raw
    else:
        ref_iv = None  # treat as missing, fall back to 20%

    # Safe default always inside [1, 300]
    _sigma_default_pct = float(np.clip(round((ref_iv or 0.20) * 100, 1), 1.0, 299.0))

    sigma_source = st.radio(
        "s para modelos",
        ["IV cadena (cercano)", "Manual"],
        help="IV cadena usa la vol implicita del strike mas cercano. Manual permite fijar sigma libremente.",
    )
    if sigma_source == "Manual":
        atm_sigma = st.number_input(
            "sigma manual (%)",
            min_value=1.0, max_value=300.0,
            value=_sigma_default_pct,
            step=0.5, format="%.1f",
        ) / 100
    else:
        atm_sigma = ref_iv if ref_iv else 0.20

# Final safety clamp: ensure atm_sigma is always in a sane range for the models
atm_sigma = float(np.clip(atm_sigma, 0.01, 3.00))

# Moneyness tag
if opt_type_live == "call":
    _mono = "ATM" if abs(chosen_strike - spot) / spot < 0.015 else ("ITM" if spot > chosen_strike else "OTM")
else:
    _mono = "ATM" if abs(chosen_strike - spot) / spot < 0.015 else ("ITM" if spot < chosen_strike else "OTM")
_mono_color = {"ATM": "#f59e0b", "ITM": "#34d399", "OTM": "#f87171"}[_mono]

# Market mid reference (closest chain row)
market_mid = df_chain.loc[ref_idx, "midPrice"] if ref_idx in df_chain.index else None
if market_mid is not None and market_mid <= 0.001:
    market_mid = None

#  Compute all models 
with st.spinner("Calculando valuación con todos los modelos..."):
    atm_results = compare_all_models(
        S=spot, K=chosen_strike, T=T_live, r=r_manual, sigma=atm_sigma, q=q_live,
        heston_params={"v0": atm_sigma**2, "kappa": 2.0, "theta": atm_sigma**2, "xi": 0.3, "rho": -0.5},
        merton_params={"lam": 1.0, "mu_j": -0.05, "sigma_j": 0.15},
    )

#  Results cards 
col_atm = st.columns(5)
with col_atm[0]:
    st.markdown(
        f"""<div style='background:{card};border:1px solid {border};border-radius:8px;padding:0.9rem 1rem'>
        <div style='font-size:0.65rem;color:{text_sub};text-transform:uppercase;letter-spacing:0.08em'>Spot / Strike</div>
        <div style='font-family:JetBrains Mono,monospace;font-size:1.05rem;color:{text_main};margin-top:0.2rem'>S = ${spot:.2f}</div>
        <div style='font-family:JetBrains Mono,monospace;font-size:1.05rem;color:{accent}'>K = ${chosen_strike:.2f}</div>
        <div style='margin-top:0.35rem'><span style='background:{_mono_color}22;color:{_mono_color};
        border:1px solid {_mono_color}44;border-radius:3px;padding:0.1rem 0.45rem;
        font-size:0.7rem;font-family:JetBrains Mono,monospace'>{_mono}</span></div>
        <div style='font-size:0.72rem;color:{text_sub};margin-top:0.35rem'>σ = {atm_sigma*100:.2f}%</div>
        </div>""",
        unsafe_allow_html=True,
    )

for col_a, (model, color) in zip(col_atm[1:], MODEL_COLORS.items()):
    price_m = atm_results[model][opt_type_live]
    delta_vs_mkt = price_m - market_mid if market_mid is not None else None
    col_a.markdown(f"""
    <div style='background:{card};border:1px solid {border};border-top:3px solid {color};border-radius:8px;padding:0.8rem;text-align:center'>
        <div style='font-size:0.65rem;color:{color};letter-spacing:0.1em;text-transform:uppercase'>{model}</div>
        <div style='font-family:JetBrains Mono,monospace;font-size:1.15rem;font-weight:600;color:{text_main}'>${price_m:.3f}</div>
        {'<div style="font-size:0.72rem;color:' + (positive if delta_vs_mkt>=0 else negative) + '">' + f"vs mkt: {delta_vs_mkt:+.3f}" + '</div>' if delta_vs_mkt is not None else ''}
    </div>
    """, unsafe_allow_html=True)

#  Info bar 
info_parts = [f"T = {T_live*365:.0f} días ({T_live:.4f} años)", f"r = {r_manual*100:.2f}%", f"σ = {atm_sigma*100:.2f}%"]
if market_mid is not None:
    info_parts.insert(0, f"Precio mkt (mid, K≈{ref_strike_label:.2f}) = ${market_mid:.3f}")
if ref_iv:
    info_parts.insert(1 if market_mid is None else 1, f"IV cadena (K={ref_strike_label:.2f}) = {ref_iv*100:.2f}%")
st.markdown(
    "<div style='background:" + card + ";border:1px solid " + border + ";border-radius:6px;"
    "padding:0.6rem 1rem;margin-top:0.6rem;font-size:0.8rem'>"
    + ("&nbsp;·&nbsp;".join(
        [f"<span style='color:{text_sub}'>{p.split(' = ')[0]} = </span>"
         f"<span style='color:{text_main};font-family:JetBrains Mono,monospace'>{p.split(' = ', 1)[1]}</span>"
         for p in info_parts]
    ))
    + "</div>",
    unsafe_allow_html=True,
)

# alias for downstream highlight logic
atm_strike = chosen_strike

st.divider()


#  Tabs 
tab1, tab2, tab3, tab4 = st.tabs(["Cadena de opciones", "Sonrisa IV de mercado", "Valuación por strike", "Vol histórica vs IV"])

with tab1:
    display_cols = ["strike", "bid", "ask", "midPrice", "volume", "openinterest", "IV_market", "impliedvolatility"]
    available_cols = [c for c in display_cols if c in df_chain.columns]
    df_display = df_chain[available_cols].copy()
    df_display.columns = [c.replace("IV_market", "IV_calc (%)").replace("impliedvolatility", "IV_yahoo").replace("midPrice", "Mid") for c in df_display.columns]

    # Style ATM row
    def highlight_atm(row):
        color_bg = f"background-color: {'#f59e0b22' if dark else '#fef3c722'}"
        if abs(row["strike"] - atm_strike) < 0.01:
            return [color_bg] * len(row)
        return [""] * len(row)

    styled = df_display.style.apply(highlight_atm, axis=1).format({
        "strike": "${:.2f}", "bid": "${:.3f}", "ask": "${:.3f}", "Mid": "${:.3f}",
        "IV_calc (%)": "{:.1f}%", "IV_yahoo": "{:.1%}",
    }, na_rep="—")

    st.dataframe(styled, use_container_width=True, height=420)
    st.caption(f"ATM strike resaltado = ${atm_strike:.2f}  ·  {len(df_chain)} strikes mostrados")

with tab2:
    df_iv = df_chain.dropna(subset=["IV_market"]).copy()

    if len(df_iv) < 3:
        st.warning("No hay suficientes datos de IV para graficar la sonrisa.")
    else:
        fig_iv = go.Figure()

        # Market IV from chain
        fig_iv.add_trace(go.Scatter(
            x=df_iv["strike"], y=df_iv["IV_market"],
            mode="markers+lines",
            name="IV mercado (bid/ask mid)",
            line=dict(color=accent, width=2),
            marker=dict(size=6),
        ))

        # Yahoo IV if available
        if "impliedvolatility" in df_chain.columns:
            df_yahoo = df_chain.dropna(subset=["impliedvolatility"]).copy()
            df_yahoo = df_yahoo[df_yahoo["impliedvolatility"] > 0]
            fig_iv.add_trace(go.Scatter(
                x=df_yahoo["strike"], y=df_yahoo["impliedvolatility"] * 100,
                mode="markers",
                name="IV Yahoo Finance",
                marker=dict(color="#a78bfa", size=5, symbol="diamond"),
            ))

        fig_iv.add_vline(x=spot, line_dash="dash", line_color=positive, annotation_text=f"S={spot:.2f}")
        fig_iv.add_vline(x=atm_strike, line_dash="dot", line_color="#f59e0b", annotation_text="ATM")
        fig_iv.update_layout(**plotly_layout(f"Sonrisa de Volatilidad Implícita — {ticker_input} · {exp_choice}"), height=400)
        fig_iv.update_xaxes(title="Strike K")
        fig_iv.update_yaxes(title="IV (%)")
        st.plotly_chart(fig_iv, use_container_width=True)

        # OI chart if available
        if "openinterest" in df_chain.columns:
            df_oi = df_chain[df_chain["openinterest"] > 0].copy()
            if not df_oi.empty:
                fig_oi = go.Figure()
                fig_oi.add_trace(go.Bar(
                    x=df_oi["strike"], y=df_oi["openinterest"],
                    name="Open Interest",
                    marker_color=[accent if abs(k - spot) / spot < 0.05 else grid_col for k in df_oi["strike"]],
                ))
                fig_oi.add_vline(x=spot, line_dash="dash", line_color=positive, annotation_text=f"S")
                fig_oi.update_layout(**plotly_layout(f"Open Interest por Strike — {opt_type_live.upper()}"), height=280)
                fig_oi.update_xaxes(title="Strike")
                fig_oi.update_yaxes(title="Open Interest")
                st.plotly_chart(fig_oi, use_container_width=True)

with tab3:
    st.markdown(f"<div style='font-size:0.82rem;color:{text_sub};margin-bottom:0.8rem'>Valuación de todos los modelos a través de la cadena de strikes disponibles (usando IV ATM como σ base).</div>", unsafe_allow_html=True)

    strikes_live = df_chain["strike"].values
    if len(strikes_live) > 40:
        # Subsample for performance
        idx_sub = np.round(np.linspace(0, len(strikes_live) - 1, 40)).astype(int)
        strikes_live = strikes_live[idx_sub]

    with st.spinner("Calculando valuación por strike..."):
        bsm_live   = [BSMEngine(spot, k, T_live, r_manual, atm_sigma, q_live).call_price()
                       if opt_type_live == "call" else
                       BSMEngine(spot, k, T_live, r_manual, atm_sigma, q_live).put_price()
                       for k in strikes_live]
        merton_live= [MertonEngine(spot, k, T_live, r_manual, atm_sigma, q_live, 1.0, -0.05, 0.15).call_price()
                       if opt_type_live == "call" else
                       MertonEngine(spot, k, T_live, r_manual, atm_sigma, q_live, 1.0, -0.05, 0.15).put_price()
                       for k in strikes_live]

    fig_chain = go.Figure()
    fig_chain.add_trace(go.Scatter(
        x=strikes_live, y=bsm_live,
        name="BSM", line=dict(color=MODEL_COLORS["BSM"], width=2, dash="dash"),
    ))
    fig_chain.add_trace(go.Scatter(
        x=strikes_live, y=merton_live,
        name="Merton", line=dict(color=MODEL_COLORS["Merton"], width=2),
    ))

    # Market bid/ask band
    df_bid_ask = df_chain[(df_chain["bid"] > 0) & (df_chain["ask"] > 0)].copy()
    if not df_bid_ask.empty:
        fig_chain.add_trace(go.Scatter(
            x=pd.concat([df_bid_ask["strike"], df_bid_ask["strike"].iloc[::-1]]),
            y=pd.concat([df_bid_ask["ask"], df_bid_ask["bid"].iloc[::-1]]),
            fill="toself",
            fillcolor=f"rgba(52,211,153,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Bid/Ask mercado",
        ))

    fig_chain.add_vline(x=spot, line_dash="dash", line_color=positive, annotation_text=f"S={spot:.2f}")
    fig_chain.update_layout(**plotly_layout(f"Modelos vs Mercado — {ticker_input} {opt_type_live.upper()} · {exp_choice}"), height=420)
    fig_chain.update_xaxes(title="Strike K")
    fig_chain.update_yaxes(title=f"Precio {opt_type_live.upper()}")
    st.plotly_chart(fig_chain, use_container_width=True)

    st.caption(f"σ base usada = {atm_sigma*100:.2f}% (IV ATM) · T = {T_live*365:.0f} días · r = {r_manual*100:.2f}%")

with tab4:
    from engine import realized_vol, iv_rv_spread
    st.markdown(f"<div style='font-size:0.85rem;color:{text_sub};margin-bottom:0.8rem'>Comparacion entre la volatilidad implicita (IV) de mercado — lo que el mercado espera — y la volatilidad realizada historica. El spread IV-RV es una proxy de la prima de riesgo de varianza.</div>", unsafe_allow_html=True)

    col_rv1, col_rv2 = st.columns([1, 3])
    with col_rv1:
        rv_window  = st.slider("Ventana RV (dias)", 5, 63, 21, key="rv_win")
        rv_period  = st.selectbox("Periodo historico", ["3mo", "6mo", "1y", "2y"], index=1, key="rv_per")
        show_spread = st.checkbox("Mostrar spread IV-RV", value=True, key="rv_sp")

    @st.cache_data(ttl=300)
    def fetch_long_history(ticker, period):
        import yfinance as yf
        import pandas as pd
        h = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = [c[0] for c in h.columns]
        return h.reset_index()

    with st.spinner(f"Cargando historico {rv_period}..."):
        hist_rv = fetch_long_history(ticker_input, rv_period)

    close_rv = None
    for col_name in ["Close", "close", "Adj Close"]:
        if col_name in hist_rv.columns:
            close_rv = hist_rv[col_name].dropna().values
            break

    if close_rv is None or len(close_rv) < rv_window + 5:
        st.warning(f"No hay suficiente historia para calcular RV con ventana {rv_window}d.")
    else:
        rv_series = realized_vol(close_rv, window=rv_window) * 100  # to %
        date_col  = None
        for dc in ["Date", "Datetime", "index"]:
            if dc in hist_rv.columns:
                date_col = hist_rv[dc].values
                break
        dates = date_col if date_col is not None else list(range(len(close_rv)))

        # ATM IV as flat line (from current chain)
        atm_iv_pct = (atm_iv_auto * 100) if atm_iv_auto else (atm_sigma * 100)

        with col_rv2:
            fig_rv = go.Figure()
            fig_rv.add_trace(go.Scatter(
                x=dates, y=rv_series,
                name=f"RV {rv_window}d",
                line=dict(color=accent, width=2),
            ))
            fig_rv.add_hline(y=atm_iv_pct, line_dash="dash",
                              line_color="#f59e0b", line_width=1.5,
                              annotation_text=f"IV ATM = {atm_iv_pct:.1f}%",
                              annotation_font_color="#f59e0b")

            if show_spread:
                rv_valid = rv_series[~np.isnan(rv_series)]
                spread_arr = atm_iv_pct - rv_valid
                dates_valid = [d for d, v in zip(dates, rv_series) if not np.isnan(v)]
                fig_rv.add_trace(go.Scatter(
                    x=dates_valid, y=spread_arr,
                    name="Spread IV-RV",
                    line=dict(color="#a78bfa", width=1.5, dash="dot"),
                    yaxis="y2",
                ))

            _rvl = dict(
                template="plotly_dark" if dark else "plotly_white",
                paper_bgcolor=paper_bg, plot_bgcolor=plot_bg,
                font=dict(family="Inter", color=text_main, size=12),
                title=dict(text=f"{ticker_input} — Volatilidad Realizada vs IV de Mercado", font=dict(size=14, color=text_main)),
                xaxis=dict(gridcolor=grid_col),
                yaxis=dict(gridcolor=grid_col, title="Volatilidad (%)"),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=40, r=60, t=40, b=40),
                height=400,
            )
            if show_spread:
                _rvl["yaxis2"] = dict(overlaying="y", side="right", title="Spread IV-RV (%)",
                                       showgrid=False, color="#a78bfa")
            fig_rv.update_layout(**_rvl)
            st.plotly_chart(fig_rv, use_container_width=True)

        # Summary stats
        rv_valid_arr = rv_series[~np.isnan(rv_series)]
        if len(rv_valid_arr) > 0:
            spread_stats = iv_rv_spread(
                [atm_iv_pct / 100] * len(rv_valid_arr),
                rv_valid_arr / 100,
            )
            st.markdown(f"<div style='font-size:0.72rem;color:{text_sub};text-transform:uppercase;letter-spacing:0.1em;margin:0.8rem 0 0.4rem'>Estadisticos del spread IV-RV</div>", unsafe_allow_html=True)
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("IV ATM actual", f"{atm_iv_pct:.2f}%")
            mc2.metric(f"RV media {rv_window}d", f"{np.nanmean(rv_series):.2f}%")
            mc3.metric("Spread medio", f"{spread_stats['mean_spread']*100:.2f}%")
            mc4.metric("% tiempo IV > RV", f"{spread_stats['pct_positive']:.1f}%")

            st.markdown(f"""
            <div style='background:{card};border:1px solid {border};border-radius:8px;padding:0.9rem 1.1rem;margin-top:0.6rem;font-size:0.83rem;line-height:1.7'>
                <strong style='color:{text_main}'>Interpretacion:</strong>
                <span style='color:{text_sub}'> Un spread IV-RV positivo ({spread_stats["mean_spread"]*100:+.2f}%) indica que el mercado paga
                una prima por proteccion de volatilidad (variance risk premium). Los vendedores de opciones
                capturan esta prima sistematicamente a largo plazo, a cambio de asumir riesgo de cola.
                Un spread negativo indicaria que las opciones estan baratas respecto a la vol realizada.</span>
            </div>
            """, unsafe_allow_html=True)
