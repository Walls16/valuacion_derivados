"""
Page 3 — Heston Stochastic Volatility Model (1993)
Semi-analytical pricing via characteristic function integration.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import HestonEngine, BSMEngine

st.set_page_config(page_title="Heston · VQD", page_icon="∂", layout="wide")

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
.param-badge {{ display: inline-block; background: {surface}; border: 1px solid {border}; border-radius: 5px; padding: 0.25rem 0.6rem; margin: 0.15rem; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {text_sub}; }}
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

st.markdown('<div class="page-eyebrow">Vol Estocástica · Página 3</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Modelo de Heston (1993)</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Volatilidad estocástica con reversión a la media. Genera sonrisa de volatilidad de forma endógena, sin suponer σ constante.</div>', unsafe_allow_html=True)

if mode == "Avanzado":
    st.markdown(f"""
    <div class="formula-box">
        <strong style="color:{text_main}">Dinámica del modelo</strong><br><br>
        dS = (r−q)·S·dt + √v·S·dW₁<br>
        dv = κ(θ − v)·dt + ξ·√v·dW₂<br>
        corr(dW₁, dW₂) = ρ<br><br>
        κ = velocidad de reversión &nbsp;·&nbsp; θ = varianza de largo plazo<br>
        ξ = vol de vol &nbsp;·&nbsp; v₀ = varianza inicial<br><br>
        <strong style="color:{text_main}">Condición de Feller:</strong> &nbsp; 2κθ > ξ² &nbsp; (varianza siempre positiva)
    </div>
    """, unsafe_allow_html=True)

st.divider()

col_in, col_out = st.columns([1, 1.6], gap="large")

with col_in:
    st.markdown(f"<div style='font-size:0.72rem;letter-spacing:0.14em;text-transform:uppercase;color:{text_sub};margin-bottom:0.8rem'>Parámetros del subyacente</div>", unsafe_allow_html=True)
    S  = st.number_input("Precio subyacente (S)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
    K  = st.number_input("Precio de ejercicio (K)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
    T  = st.number_input("Tiempo al vencimiento (T, años)", value=1.0, min_value=0.01, max_value=10.0, step=0.05, format="%.3f")
    r  = st.number_input("Tasa libre de riesgo (r, %)", value=5.0, step=0.25, format="%.2f") / 100
    q  = st.number_input("Dividendo continuo (q, %)", value=0.0, step=0.1, format="%.2f") / 100

    st.markdown(f"<div style='font-size:0.72rem;letter-spacing:0.14em;text-transform:uppercase;color:{text_sub};margin:1rem 0 0.5rem 0'>Parámetros de Heston</div>", unsafe_allow_html=True)
    v0    = (st.number_input("Varianza inicial (v₀ = σ₀²)", value=4.0, min_value=0.01, max_value=100.0, step=0.5, format="%.2f") / 100) ** 0 * \
             (st.number_input("σ₀ inicial (%)", value=20.0, min_value=1.0, max_value=100.0, step=0.5, format="%.2f") / 100) ** 2
    kappa = st.slider("κ — velocidad de reversión", 0.1, 10.0, 2.0, 0.1)
    theta_pct = st.slider("θ — vol largo plazo (%)", 1.0, 80.0, 20.0, 0.5)
    theta = (theta_pct / 100) ** 2
    xi    = st.slider("ξ — vol de vol", 0.05, 1.5, 0.3, 0.05)
    rho   = st.slider("ρ — correlación S-v", -0.99, 0.99, -0.7, 0.01)

    feller = 2 * kappa * theta > xi**2
    feller_color = "#34d399" if feller else "#f87171"
    st.markdown(f"""
    <div style='background:{card};border:1px solid {feller_color}44;border-radius:6px;padding:0.6rem 1rem;margin-top:0.5rem;font-size:0.8rem'>
        <span style='color:{feller_color}'>{"✓" if feller else "✗"} Condición de Feller: 2κθ = {2*kappa*theta:.4f} {'>' if feller else '<'} ξ² = {xi**2:.4f}</span>
    </div>
    """, unsafe_allow_html=True)

# Compute
with st.spinner("Calculando Heston (integración numérica)..."):
    h = HestonEngine(S, K, T, r, q, v0, kappa, theta, xi, rho)
    heston_call = h.call_price()
    heston_put  = h.put_price()

bsm_sigma = np.sqrt(v0)
bsm = BSMEngine(S, K, T, r, bsm_sigma, q)
bsm_call = bsm.call_price()
bsm_put  = bsm.put_price()

with col_out:
    st.markdown(f"<div style='font-size:0.72rem;letter-spacing:0.14em;text-transform:uppercase;color:{text_sub};margin-bottom:0.8rem'>Resultados</div>", unsafe_allow_html=True)

    rc, rp = st.columns(2)
    diff_c = heston_call - bsm_call
    diff_p = heston_put  - bsm_put

    with rc:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Call — Heston</div>
            <div class="result-value">${heston_call:.4f}</div>
            <div style='font-size:0.78rem;color:{"#34d399" if diff_c>=0 else "#f87171"};margin-top:0.3rem'>
                vs BSM(σ₀): {"+" if diff_c>=0 else ""}{diff_c:.4f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with rp:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Put — Heston</div>
            <div class="result-value">${heston_put:.4f}</div>
            <div style='font-size:0.78rem;color:{"#34d399" if diff_p>=0 else "#f87171"};margin-top:0.3rem'>
                vs BSM(σ₀): {"+" if diff_p>=0 else ""}{diff_p:.4f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    m1, m2, m3 = st.columns(3)
    m1.metric("BSM Call (σ₀)", f"${bsm_call:.4f}")
    m2.metric("Heston − BSM", f"${diff_c:.4f}", delta_color="normal")
    m3.metric("σ₀ equivalente", f"{np.sqrt(v0)*100:.2f}%")

st.divider()

tab1, tab2, tab3 = st.tabs(["Sonrisa de volatilidad", "Superficie IV", "Sensibilidad a ρ y ξ"])

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
    with st.spinner("Calculando sonrisa Heston..."):
        ivs_heston = h.vol_smile(strikes_smile)
    ivs_bsm_flat = np.full(len(strikes_smile), np.sqrt(v0))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strikes_smile, y=ivs_heston * 100, name="Heston IV", line=dict(color=accent, width=2.5)))
    fig.add_trace(go.Scatter(x=strikes_smile, y=ivs_bsm_flat * 100, name="BSM Flat IV", line=dict(color="#f87171", width=1.5, dash="dash")))
    fig.add_vline(x=K, line_dash="dot", line_color=text_sub, annotation_text="K")
    fig.update_layout(**plotly_layout("Sonrisa de Volatilidad — Heston vs BSM"), height=380)
    fig.update_xaxes(title="Strike K")
    fig.update_yaxes(title="Volatilidad Implícita (%)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    maturities = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    strikes_surf = np.linspace(S * 0.75, S * 1.25, 25)

    if mode == "Avanzado":
        st.info("Calculando superficie completa (puede tomar ~15s)...")
        surface_iv = np.zeros((len(maturities), len(strikes_surf)))
        with st.spinner("Calculando superficie IV..."):
            for i, t_mat in enumerate(maturities):
                h_t = HestonEngine(S, K, t_mat, r, q, v0, kappa, theta, xi, rho)
                surface_iv[i] = h_t.vol_smile(strikes_surf) * 100

        fig_surf = go.Figure(data=[go.Surface(
            z=surface_iv, x=strikes_surf, y=maturities,
            colorscale="Blues" if dark else "RdBu",
            showscale=True,
        )])
        fig_surf.update_layout(
            paper_bgcolor=paper_bg,
            font=dict(color=text_main),
            scene=dict(
                xaxis=dict(title="Strike", backgroundcolor=plot_bg, gridcolor=grid_col),
                yaxis=dict(title="Madurez (años)", backgroundcolor=plot_bg, gridcolor=grid_col),
                zaxis=dict(title="IV (%)", backgroundcolor=plot_bg, gridcolor=grid_col),
            ),
            title=dict(text="Superficie de Volatilidad Implícita — Heston", font=dict(color=text_main, size=14)),
            height=500,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_surf, use_container_width=True)
    else:
        st.info("Activa el **Modo Avanzado** en la barra lateral para ver la superficie 3D de volatilidad.")

with tab3:
    rhos  = np.linspace(-0.95, 0.95, 30)
    xis   = np.linspace(0.05, 1.2, 30)

    col_r, col_xi = st.columns(2)

    with col_r:
        with st.spinner("Calculando sensibilidad a ρ..."):
            calls_rho = []
            for rho_val in rhos:
                he = HestonEngine(S, K, T, r, q, v0, kappa, theta, xi, rho_val)
                calls_rho.append(he.call_price())
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=rhos, y=calls_rho, line=dict(color=accent, width=2), name="Call Heston"))
        fig_r.add_vline(x=rho, line_dash="dot", line_color=text_sub, annotation_text="ρ actual")
        fig_r.update_layout(**plotly_layout("Precio Call vs ρ"), height=300)
        fig_r.update_xaxes(title="ρ (correlación S-v)")
        fig_r.update_yaxes(title="Precio Call")
        st.plotly_chart(fig_r, use_container_width=True)

    with col_xi:
        with st.spinner("Calculando sensibilidad a ξ..."):
            calls_xi = []
            for xi_val in xis:
                he = HestonEngine(S, K, T, r, q, v0, kappa, theta, xi_val, rho)
                calls_xi.append(he.call_price())
        fig_x = go.Figure()
        fig_x.add_trace(go.Scatter(x=xis, y=calls_xi, line=dict(color="#a78bfa", width=2), name="Call Heston"))
        fig_x.add_vline(x=xi, line_dash="dot", line_color=text_sub, annotation_text="ξ actual")
        fig_x.update_layout(**plotly_layout("Precio Call vs ξ"), height=300)
        fig_x.update_xaxes(title="ξ (vol de vol)")
        fig_x.update_yaxes(title="Precio Call")
        st.plotly_chart(fig_x, use_container_width=True)
