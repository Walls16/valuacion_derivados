"""
Page 4 — Merton Jump-Diffusion Model (1976)
Poisson jump process superimposed on GBM.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import MertonEngine, BSMEngine

st.set_page_config(page_title="Merton · VQD", page_icon="∂", layout="wide")

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
.page-title {{ font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 700; color: {text_main}; }}
.page-eyebrow {{ font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.16em; color: {accent}; text-transform: uppercase; margin-bottom: 0.5rem; }}
.page-sub {{ font-size: 0.9rem; color: {text_sub}; margin-bottom: 1.5rem; }}
.formula-box {{ background: {card}; border: 1px solid {border}; border-left: 3px solid {accent}; border-radius: 8px; padding: 1.2rem 1.4rem; margin: 1rem 0; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: {text_sub}; line-height: 1.9; }}
.result-card {{ background: {card}; border: 1px solid {border}; border-radius: 10px; padding: 1.3rem 1.5rem; }}
.result-value {{ font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 600; color: {text_main}; }}
.result-label {{ font-size: 0.72rem; color: {text_sub}; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.2rem; }}
[data-testid="metric-container"] {{ background: {card} !important; border: 1px solid {border} !important; border-radius: 8px !important; padding: 0.8rem 1rem !important; }}
[data-testid="metric-container"] label {{ color: {text_sub} !important; font-size: 0.75rem !important; }}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{ color: {text_main} !important; font-family: 'JetBrains Mono', monospace !important; }}
.stTabs [data-baseweb="tab"] {{ color: {text_sub} !important; }}
.stTabs [aria-selected="true"] {{ color: {accent} !important; border-bottom-color: {accent} !important; }}
label {{ color: {text_sub} !important; font-size: 0.82rem !important; }}
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
    mode = st.radio("Modo de visualización", ["Básico", "Avanzado"], index=0)

st.markdown('<div class="page-eyebrow">Jump-Diffusion · Página 4</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Modelo de Merton (1976)</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Proceso difusivo con saltos de Poisson log-normales. Captura eventos de cola, skew de volatilidad y discontinuidades de precio.</div>', unsafe_allow_html=True)

if mode == "Avanzado":
    st.markdown(f"""
    <div class="formula-box">
        <strong style="color:{text_main}">Dinámica del proceso</strong><br><br>
        dS/S = (r − λκ)·dt + σ·dW + (J−1)·dN<br><br>
        N ~ Poisson(λ) &nbsp;·&nbsp; λ = intensidad de saltos (saltos/año)<br>
        ln(J) ~ N(μⱼ, σⱼ²) &nbsp;·&nbsp; κ = E[J−1] = e<sup>μⱼ + σⱼ²/2</sup> − 1<br><br>
        <strong style="color:{text_main}">Precio (serie de Poisson)</strong><br><br>
        C = Σ<sub>n=0</sub><sup>∞</sup> e<sup>−λ'T</sup>(λ'T)<sup>n</sup>/n! · BSM(S, K, T, rₙ, σₙ)<br><br>
        rₙ = r − λκ + n·μⱼ/T &nbsp;·&nbsp; σₙ = √(σ² + n·σⱼ²/T)
    </div>
    """, unsafe_allow_html=True)

st.divider()

col_in, col_out = st.columns([1, 1.6], gap="large")

with col_in:
    st.markdown(f"<div style='font-size:0.72rem;letter-spacing:0.14em;text-transform:uppercase;color:{text_sub};margin-bottom:0.8rem'>Parámetros del subyacente</div>", unsafe_allow_html=True)
    S     = st.number_input("Precio subyacente (S)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
    K     = st.number_input("Precio de ejercicio (K)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
    T     = st.number_input("Tiempo al vencimiento (T, años)", value=1.0, min_value=0.01, max_value=10.0, step=0.05, format="%.3f")
    r     = st.number_input("Tasa libre de riesgo (r, %)", value=5.0, step=0.25, format="%.2f") / 100
    sigma = st.number_input("Volatilidad difusiva (σ, %)", value=15.0, min_value=0.1, max_value=200.0, step=0.5, format="%.2f") / 100
    q     = st.number_input("Dividendo continuo (q, %)", value=0.0, step=0.1, format="%.2f") / 100

    st.markdown(f"<div style='font-size:0.72rem;letter-spacing:0.14em;text-transform:uppercase;color:{text_sub};margin:1rem 0 0.5rem 0'>Parámetros de salto</div>", unsafe_allow_html=True)
    lam     = st.slider("λ — intensidad de saltos (por año)", 0.0, 10.0, 1.0, 0.1)
    mu_j    = st.slider("μⱼ — tamaño medio del salto (log)", -1.0, 1.0, -0.1, 0.01)
    sigma_j = st.slider("σⱼ — volatilidad del salto", 0.01, 1.0, 0.15, 0.01)
    n_terms = st.slider("Términos de la serie", 10, 100, 50, 5) if mode == "Avanzado" else 50

    kappa_j = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    mean_jump_pct = kappa_j * 100

    st.markdown(f"""
    <div style='background:{card};border:1px solid {border};border-radius:6px;padding:0.7rem 1rem;margin-top:0.5rem;font-size:0.8rem'>
        <span style='color:{text_sub}'>Salto medio E[J−1] = </span>
        <span style='font-family:JetBrains Mono,monospace;color:{"#34d399" if kappa_j>=0 else "#f87171"}'>
            {"+" if kappa_j>=0 else ""}{mean_jump_pct:.2f}%
        </span><br>
        <span style='color:{text_sub}'>Frecuencia esperada: </span>
        <span style='font-family:JetBrains Mono,monospace;color:{text_main}'>{lam:.1f} saltos/año</span>
    </div>
    """, unsafe_allow_html=True)

# Compute
merton = MertonEngine(S, K, T, r, sigma, q, lam, mu_j, sigma_j, n_terms)
merton_call = merton.call_price()
merton_put  = merton.put_price()

bsm_total_sigma = np.sqrt(sigma**2 + lam * sigma_j**2)
bsm = BSMEngine(S, K, T, r, bsm_total_sigma, q)
bsm_call = bsm.call_price()
bsm_pure = BSMEngine(S, K, T, r, sigma, q)
bsm_nodiff = bsm_pure.call_price()

with col_out:
    st.markdown(f"<div style='font-size:0.72rem;letter-spacing:0.14em;text-transform:uppercase;color:{text_sub};margin-bottom:0.8rem'>Resultados</div>", unsafe_allow_html=True)

    rc, rp = st.columns(2)
    diff_c = merton_call - bsm_nodiff

    with rc:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Call — Merton Jump-Diff</div>
            <div class="result-value">${merton_call:.4f}</div>
            <div style='font-size:0.78rem;color:{"#34d399" if diff_c>=0 else "#f87171"};margin-top:0.3rem'>
                vs BSM(σ): {"+" if diff_c>=0 else ""}{diff_c:.4f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with rp:
        diff_p = merton_put - BSMEngine(S, K, T, r, sigma, q).put_price()
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Put — Merton Jump-Diff</div>
            <div class="result-value">${merton_put:.4f}</div>
            <div style='font-size:0.78rem;color:{"#34d399" if diff_p>=0 else "#f87171"};margin-top:0.3rem'>
                vs BSM(σ): {"+" if diff_p>=0 else ""}{diff_p:.4f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    m1, m2, m3 = st.columns(3)
    jump_premium = merton_call - bsm_nodiff
    m1.metric("BSM Call (σ difusiva)", f"${bsm_nodiff:.4f}")
    m2.metric("Prima por saltos", f"${jump_premium:.4f}")
    m3.metric("% de la prima total", f"{jump_premium/merton_call*100:.1f}%")

st.divider()

tab1, tab2, tab3 = st.tabs(["Sonrisa de volatilidad", "Efecto de λ y σⱼ", "Trayectorias simuladas"])

def plotly_layout(title=""):
    return dict(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor=paper_bg, plot_bgcolor=plot_bg,
        font=dict(family="Inter", color=text_main, size=12),
        title=dict(text=title, font=dict(size=14, color=text_main)),
        xaxis=dict(gridcolor=grid_col),
        yaxis=dict(gridcolor=grid_col),
        margin=dict(l=40, r=20, t=40, b=40),
    )

with tab1:
    strikes_smile = np.linspace(S * 0.7, S * 1.3, 50)
    ivs_merton = merton.vol_smile(strikes_smile)
    ivs_bsm    = np.full(len(strikes_smile), sigma)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strikes_smile, y=ivs_merton * 100, name="Merton IV",
                              line=dict(color=accent, width=2.5)))
    fig.add_trace(go.Scatter(x=strikes_smile, y=ivs_bsm * 100, name="BSM Flat (σ difusiva)",
                              line=dict(color="#f87171", width=1.5, dash="dash")))
    fig.add_vline(x=K, line_dash="dot", line_color=text_sub, annotation_text="K")
    fig.update_layout(**plotly_layout("Sonrisa de Volatilidad — Merton Jump-Diffusion"), height=380)
    fig.update_xaxes(title="Strike K")
    fig.update_yaxes(title="Volatilidad Implícita (%)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    col_l, col_s = st.columns(2)

    with col_l:
        lam_range = np.linspace(0, 8, 40)
        calls_lam = [MertonEngine(S, K, T, r, sigma, q, l, mu_j, sigma_j).call_price() for l in lam_range]
        fig_l = go.Figure()
        fig_l.add_trace(go.Scatter(x=lam_range, y=calls_lam, line=dict(color=accent, width=2)))
        fig_l.add_vline(x=lam, line_dash="dot", line_color=text_sub, annotation_text="λ actual")
        fig_l.add_hline(y=bsm_nodiff, line_dash="dot", line_color="#f87171", annotation_text="BSM")
        fig_l.update_layout(**plotly_layout("Call vs Intensidad λ"), height=300)
        fig_l.update_xaxes(title="λ (saltos/año)")
        fig_l.update_yaxes(title="Precio Call")
        st.plotly_chart(fig_l, use_container_width=True)

    with col_s:
        sj_range = np.linspace(0.01, 0.8, 40)
        calls_sj = [MertonEngine(S, K, T, r, sigma, q, lam, mu_j, sj).call_price() for sj in sj_range]
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=sj_range * 100, y=calls_sj, line=dict(color="#a78bfa", width=2)))
        fig_s.add_vline(x=sigma_j * 100, line_dash="dot", line_color=text_sub, annotation_text="σⱼ actual")
        fig_s.update_layout(**plotly_layout("Call vs Vol de salto σⱼ"), height=300)
        fig_s.update_xaxes(title="σⱼ (%)")
        fig_s.update_yaxes(title="Precio Call")
        st.plotly_chart(fig_s, use_container_width=True)

with tab3:
    st.markdown(f"<div style='font-size:0.8rem;color:{text_sub};margin-bottom:0.8rem'>Simulación de Monte Carlo de trayectorias con saltos de Poisson</div>", unsafe_allow_html=True)

    n_paths = st.slider("Número de trayectorias", 5, 50, 20)
    n_steps_sim = 252
    dt_sim = T / n_steps_sim
    np.random.seed(42)

    fig_mc = go.Figure()
    times = np.linspace(0, T, n_steps_sim + 1)

    for i in range(n_paths):
        S_path = [S]
        for _ in range(n_steps_sim):
            s_curr = S_path[-1]
            diffusion = (r - q - lam * kappa_j - 0.5 * sigma**2) * dt_sim + sigma * np.sqrt(dt_sim) * np.random.randn()
            n_jumps = np.random.poisson(lam * dt_sim)
            jump = sum(np.random.normal(mu_j, sigma_j) for _ in range(n_jumps))
            S_path.append(s_curr * np.exp(diffusion + jump))

        color = f"rgba(79,142,247,{0.3 + 0.4 * (i / n_paths)})" if dark else f"rgba(37,99,235,{0.3 + 0.4 * (i / n_paths)})"
        fig_mc.add_trace(go.Scatter(x=times, y=S_path, mode="lines",
                                     line=dict(width=1, color=color), showlegend=False))

    fig_mc.add_hline(y=K, line_dash="dot", line_color="#f87171", annotation_text="K")
    fig_mc.add_hline(y=S, line_dash="dot", line_color="#34d399", annotation_text="S₀")
    fig_mc.update_layout(**plotly_layout(f"Trayectorias simuladas — Merton (λ={lam}, μⱼ={mu_j}, σⱼ={sigma_j})"), height=400)
    fig_mc.update_xaxes(title="Tiempo (años)")
    fig_mc.update_yaxes(title="Precio S(t)")
    st.plotly_chart(fig_mc, use_container_width=True)
