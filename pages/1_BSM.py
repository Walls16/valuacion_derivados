"""
Page 1 — Black-Scholes-Merton
Analytical closed-form pricing for European options.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import BSMEngine, intrinsic_time_value, parity_check

st.set_page_config(page_title="BSM · VQD", page_icon="∂", layout="wide")

# ── Theme ──
dark = st.session_state.get("dark_mode", True)

bg        = "#0d0f14" if dark else "#f4f6fb"
surface   = "#131720" if dark else "#ffffff"
card      = "#1a1f2e" if dark else "#ffffff"
border    = "#2a3040" if dark else "#dde3ef"
text_main = "#e8eaf0" if dark else "#1a2035"
text_sub  = "#8892a4" if dark else "#6b7a99"
accent    = "#4f8ef7" if dark else "#2563eb"
grid_col  = "#1e2535" if dark else "#e8edf5"
plot_bg   = "#131720" if dark else "#ffffff"
paper_bg  = "#0d0f14" if dark else "#f4f6fb"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Playfair+Display:wght@700&display=swap');
html, body, [data-testid="stApp"] {{ background-color: {bg} !important; color: {text_main} !important; font-family: 'Inter', sans-serif; }}
[data-testid="stSidebar"] {{ background-color: {surface} !important; border-right: 1px solid {border} !important; }}
[data-testid="stSidebar"] * {{ color: {text_main} !important; }}
.main .block-container {{ padding-top: 1.5rem; max-width: 1100px; }}
.page-title {{ font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 700; color: {text_main}; margin-bottom: 0.2rem; }}
.page-eyebrow {{ font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.16em; color: {accent}; text-transform: uppercase; margin-bottom: 0.5rem; }}
.page-sub {{ font-size: 0.9rem; color: {text_sub}; margin-bottom: 1.5rem; }}
.formula-box {{ background: {card}; border: 1px solid {border}; border-left: 3px solid {accent}; border-radius: 8px; padding: 1.2rem 1.4rem; margin: 1rem 0; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: {text_sub}; line-height: 1.9; }}
.result-card {{ background: {card}; border: 1px solid {border}; border-radius: 10px; padding: 1.3rem 1.5rem; margin-bottom: 1rem; }}
.result-label {{ font-size: 0.72rem; color: {text_sub}; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.2rem; }}
.result-value {{ font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 600; color: {text_main}; }}
.result-sub {{ font-size: 0.78rem; color: {text_sub}; margin-top: 0.2rem; }}
.greek-chip {{ display: inline-block; background: {surface}; border: 1px solid {border}; border-radius: 5px; padding: 0.3rem 0.7rem; margin: 0.2rem; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: {text_main}; }}
[data-testid="metric-container"] {{ background: {card} !important; border: 1px solid {border} !important; border-radius: 8px !important; padding: 0.8rem 1rem !important; }}
[data-testid="metric-container"] label {{ color: {text_sub} !important; font-size: 0.75rem !important; }}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{ color: {text_main} !important; font-family: 'JetBrains Mono', monospace !important; }}
.stTabs [data-baseweb="tab"] {{ color: {text_sub} !important; }}
.stTabs [aria-selected="true"] {{ color: {accent} !important; border-bottom-color: {accent} !important; }}
label {{ color: {text_sub} !important; font-size: 0.82rem !important; }}
.stToggle label {{ color: {text_main} !important; }}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("### ∂ VQD")
    st.caption("Valuación Cuantitativa de Derivados")
    st.divider()
    dark_toggle = st.toggle("Modo oscuro", value=dark, key="theme_toggle")
    if dark_toggle != dark:
        st.session_state.dark_mode = dark_toggle
        st.rerun()
    st.divider()
    mode = st.radio("Modo de visualización", ["Básico", "Avanzado"], index=0)

# ── Header ──
st.markdown('<div class="page-eyebrow">Modelo Analítico · Página 1</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Black-Scholes-Merton</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Fórmula cerrada para la valuación de opciones europeas bajo volatilidad constante.</div>', unsafe_allow_html=True)

if mode == "Avanzado":
    st.markdown(f"""
    <div class="formula-box">
        <strong style="color:{text_main}">Fórmulas BSM</strong><br><br>
        C = S·e<sup>-qT</sup>·N(d₁) − K·e<sup>-rT</sup>·N(d₂)<br>
        P = K·e<sup>-rT</sup>·N(−d₂) − S·e<sup>-qT</sup>·N(−d₁)<br><br>
        d₁ = [ln(S/K) + (r − q + σ²/2)·T] / (σ√T)<br>
        d₂ = d₁ − σ√T<br><br>
        <span style="color:{text_sub}">donde N(·) es la CDF de la normal estándar.</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Inputs ──
col_in, col_out = st.columns([1, 1.6], gap="large")

with col_in:
    st.markdown(f"<div style='font-size:0.72rem;letter-spacing:0.14em;text-transform:uppercase;color:{text_sub};margin-bottom:0.8rem'>Parámetros</div>", unsafe_allow_html=True)

    S     = st.number_input("Precio del subyacente (S)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
    K     = st.number_input("Precio de ejercicio (K)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
    T     = st.number_input("Tiempo al vencimiento (T, años)", value=1.0, min_value=0.01, max_value=10.0, step=0.05, format="%.3f")
    r     = st.number_input("Tasa libre de riesgo (r, %)", value=5.0, min_value=0.0, max_value=30.0, step=0.25, format="%.2f") / 100
    sigma = st.number_input("Volatilidad implícita (σ, %)", value=20.0, min_value=0.1, max_value=200.0, step=0.5, format="%.2f") / 100
    q     = st.number_input("Dividendo continuo (q, %)", value=0.0, min_value=0.0, max_value=20.0, step=0.1, format="%.2f") / 100

    if mode == "Avanzado":
        st.markdown(f"""
        <div style='background:{card};border:1px solid {border};border-radius:6px;padding:0.8rem 1rem;margin-top:0.5rem;font-family:JetBrains Mono,monospace;font-size:0.78rem;color:{text_sub}'>
        d₁ = {(np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T)):.4f}<br>
        d₂ = {(np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T)) - sigma*np.sqrt(T):.4f}
        </div>
        """, unsafe_allow_html=True)

# ── Compute ──
bsm = BSMEngine(S, K, T, r, sigma, q)
call = bsm.call_price()
put  = bsm.put_price()
greeks = bsm.greeks()
iv_dec = intrinsic_time_value(call, S, K, "call")
parity = parity_check(call, put, S, K, T, r, q)

with col_out:
    st.markdown(f"<div style='font-size:0.72rem;letter-spacing:0.14em;text-transform:uppercase;color:{text_sub};margin-bottom:0.8rem'>Resultados</div>", unsafe_allow_html=True)

    rc, rp = st.columns(2)
    with rc:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Call Europeo</div>
            <div class="result-value">${call:.4f}</div>
            <div class="result-sub">Intrínseco: ${iv_dec['intrinsic']:.4f} · Temporal: ${iv_dec['time_value']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with rp:
        iv_put = intrinsic_time_value(put, S, K, "put")
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Put Europeo</div>
            <div class="result-value">${put:.4f}</div>
            <div class="result-sub">Intrínseco: ${iv_put['intrinsic']:.4f} · Temporal: ${iv_put['time_value']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    if mode == "Avanzado":
        st.markdown(f"""
        <div style='background:{card};border:1px solid {border};border-radius:8px;padding:0.8rem 1rem;font-size:0.8rem;margin-bottom:0.8rem'>
            <span style='color:{text_sub}'>Paridad Put-Call · residuo: </span>
            <span style='font-family:JetBrains Mono,monospace;color:{"#34d399" if parity["residual"]<0.001 else "#f87171"}'>
                {parity["residual"]:.6f}
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Moneyness tag
    moneyness = "ATM" if abs(S - K) / K < 0.02 else ("ITM" if S > K else "OTM")
    color_m = {"ATM": "#f59e0b", "ITM": "#34d399", "OTM": "#f87171"}[moneyness]
    st.markdown(f"<span style='background:{color_m}22;color:{color_m};border:1px solid {color_m}44;border-radius:4px;padding:0.2rem 0.6rem;font-size:0.78rem;font-family:JetBrains Mono,monospace'>{moneyness} · S/K = {S/K:.3f}</span>", unsafe_allow_html=True)

st.divider()

# ── Tabs: charts ──
tab1, tab2, tab3 = st.tabs(["Perfil de payoff", "Sensibilidad al parámetro", "Griegas"])

def plotly_layout(title=""):
    return dict(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family="Inter", color=text_main, size=12),
        title=dict(text=title, font=dict(size=14, color=text_main)),
        xaxis=dict(gridcolor=grid_col, showgrid=True),
        yaxis=dict(gridcolor=grid_col, showgrid=True),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=border),
        margin=dict(l=40, r=20, t=40, b=40),
    )

with tab1:
    strikes = np.linspace(max(S * 0.5, 1), S * 1.5, 200)
    calls_v = [BSMEngine(S, k, T, r, sigma, q).call_price() for k in strikes]
    puts_v  = [BSMEngine(S, k, T, r, sigma, q).put_price()  for k in strikes]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strikes, y=calls_v, name="Call", line=dict(color=accent, width=2)))
    fig.add_trace(go.Scatter(x=strikes, y=puts_v,  name="Put",  line=dict(color="#a78bfa", width=2)))
    fig.add_vline(x=K, line_dash="dot", line_color=text_sub, annotation_text="K", annotation_font_color=text_sub)
    fig.add_vline(x=S, line_dash="dot", line_color="#34d399", annotation_text="S", annotation_font_color="#34d399")
    fig.update_layout(**plotly_layout("Valor de la opción vs Strike"), height=360)
    fig.update_xaxes(title="Strike (K)")
    fig.update_yaxes(title="Precio")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    param_choice = st.selectbox("Variar parámetro", ["Precio subyacente (S)", "Volatilidad (σ)", "Tiempo (T)", "Tasa (r)"])

    if param_choice == "Precio subyacente (S)":
        xs = np.linspace(S * 0.5, S * 1.5, 200)
        calls_p = [BSMEngine(x, K, T, r, sigma, q).call_price() for x in xs]
        puts_p  = [BSMEngine(x, K, T, r, sigma, q).put_price()  for x in xs]
        xlabel = "Precio subyacente (S)"
    elif param_choice == "Volatilidad (σ)":
        xs = np.linspace(0.01, 1.0, 200)
        calls_p = [BSMEngine(S, K, T, r, x, q).call_price() for x in xs]
        puts_p  = [BSMEngine(S, K, T, r, x, q).put_price()  for x in xs]
        xlabel = "Volatilidad σ"
    elif param_choice == "Tiempo (T)":
        xs = np.linspace(0.02, 3.0, 200)
        calls_p = [BSMEngine(S, K, x, r, sigma, q).call_price() for x in xs]
        puts_p  = [BSMEngine(S, K, x, r, sigma, q).put_price()  for x in xs]
        xlabel = "Tiempo al vencimiento T (años)"
    else:
        xs = np.linspace(0.001, 0.20, 200)
        calls_p = [BSMEngine(S, K, T, x, sigma, q).call_price() for x in xs]
        puts_p  = [BSMEngine(S, K, T, x, sigma, q).put_price()  for x in xs]
        xlabel = "Tasa libre de riesgo r"

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=xs, y=calls_p, name="Call", line=dict(color=accent, width=2)))
    fig2.add_trace(go.Scatter(x=xs, y=puts_p,  name="Put",  line=dict(color="#a78bfa", width=2)))
    fig2.update_layout(**plotly_layout(f"Sensibilidad: {param_choice}"), height=360)
    fig2.update_xaxes(title=xlabel)
    fig2.update_yaxes(title="Precio BSM")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    g = greeks
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown(f"<div style='font-size:0.72rem;color:{text_sub};text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.6rem'>Griegas de 1.er orden</div>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Δ Call (Delta)", f"{g['call_delta']:.4f}")
        m2.metric("Δ Put (Delta)",  f"{g['put_delta']:.4f}")
        m1, m2 = st.columns(2)
        m1.metric("Θ Call (Theta/día)", f"{g['call_theta']:.4f}")
        m2.metric("Θ Put (Theta/día)",  f"{g['put_theta']:.4f}")
        m1, m2 = st.columns(2)
        m1.metric("ν Vega (por 1%)",    f"{g['vega']:.4f}")
        m2.metric("ρ Call (Rho/1%)",    f"{g['call_rho']:.4f}")

    with col_g2:
        st.markdown(f"<div style='font-size:0.72rem;color:{text_sub};text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.6rem'>Griegas de 2.° orden</div>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Γ Gamma",  f"{g['gamma']:.6f}")
        m2.metric("Vanna",    f"{g['vanna']:.4f}")
        m1, m2 = st.columns(2)
        m1.metric("Vomma",    f"{g['volga']:.4f}")
        m2.metric("Charm Call", f"{g['call_charm']:.6f}")

    if mode == "Avanzado":
        # Greek sensitivity chart
        st.markdown(f"<div style='font-size:0.72rem;color:{text_sub};text-transform:uppercase;letter-spacing:0.1em;margin:1rem 0 0.4rem 0'>Delta vs precio subyacente</div>", unsafe_allow_html=True)
        xs_d = np.linspace(S * 0.5, S * 1.5, 200)
        call_deltas = [BSMEngine(x, K, T, r, sigma, q).greeks()["call_delta"] for x in xs_d]
        put_deltas  = [BSMEngine(x, K, T, r, sigma, q).greeks()["put_delta"]  for x in xs_d]
        gammas      = [BSMEngine(x, K, T, r, sigma, q).greeks()["gamma"]      for x in xs_d]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=xs_d, y=call_deltas, name="Δ Call", line=dict(color=accent, width=2)))
        fig3.add_trace(go.Scatter(x=xs_d, y=put_deltas,  name="Δ Put",  line=dict(color="#a78bfa", width=2)))
        fig3.add_trace(go.Scatter(x=xs_d, y=gammas,      name="Γ Gamma", line=dict(color="#34d399", width=2, dash="dash"), yaxis="y2"))
        fig3.update_layout(
            **plotly_layout("Delta y Gamma vs S"),
            height=320,
            yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Gamma", color="#34d399"),
        )
        fig3.update_xaxes(title="S")
        fig3.update_yaxes(title="Delta")
        st.plotly_chart(fig3, use_container_width=True)
