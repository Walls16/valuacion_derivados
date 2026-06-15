"""
Page 6 — Comparativa de Modelos
Side-by-side pricing divergence across BSM, CRR, Heston, Merton.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import compare_all_models, BSMEngine, CRREngine, HestonEngine, MertonEngine

st.set_page_config(page_title="Comparativa · VQD", page_icon="∂", layout="wide")

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
.compare-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem; margin: 1rem 0; }}
.model-result-card {{ background: {card}; border: 1px solid {border}; border-radius: 10px; padding: 1.1rem 1.2rem; text-align: center; }}
.model-name {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem; }}
.model-price {{ font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 600; color: {text_main}; }}
.model-delta {{ font-size: 0.75rem; margin-top: 0.2rem; }}
[data-testid="metric-container"] {{ background: {card} !important; border: 1px solid {border} !important; border-radius: 8px !important; padding: 0.8rem 1rem !important; }}
[data-testid="metric-container"] label {{ color: {text_sub} !important; font-size: 0.75rem !important; }}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{ color: {text_main} !important; font-family: 'JetBrains Mono', monospace !important; }}
.stTabs [data-baseweb="tab"] {{ color: {text_sub} !important; }}
.stTabs [aria-selected="true"] {{ color: {accent} !important; border-bottom-color: {accent} !important; }}
label {{ color: {text_sub} !important; font-size: 0.82rem !important; }}
.section-header {{ font-size: 0.7rem; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.16em; text-transform: uppercase; color: {text_sub}; margin: 1.5rem 0 0.8rem 0; }}
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

st.markdown('<div class="page-eyebrow">Benchmark · Página 6</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Comparativa de Modelos</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Divergencia de precios y sensibilidades entre BSM, CRR, Heston y Merton bajo los mismos parámetros base. BSM es el benchmark de referencia.</div>', unsafe_allow_html=True)
st.divider()

# ── Inputs ──
with st.expander("⚙️ Parámetros comunes", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        S     = st.number_input("S — precio subyacente", value=100.0, step=1.0, format="%.2f")
        K     = st.number_input("K — precio de ejercicio", value=100.0, step=1.0, format="%.2f")
    with c2:
        T     = st.number_input("T — tiempo (años)", value=1.0, min_value=0.01, step=0.05, format="%.3f")
        r     = st.number_input("r — tasa (%)", value=5.0, step=0.25, format="%.2f") / 100
    with c3:
        sigma = st.number_input("σ — volatilidad base (%)", value=20.0, min_value=0.1, step=0.5, format="%.2f") / 100
        q     = st.number_input("q — dividendo (%)", value=0.0, step=0.1, format="%.2f") / 100

with st.expander("⚙️ Parámetros de Heston y Merton", expanded=False):
    ch, cm = st.columns(2)
    with ch:
        st.markdown(f"<div style='font-size:0.72rem;color:{text_sub};margin-bottom:0.4rem'>Heston</div>", unsafe_allow_html=True)
        h_kappa = st.slider("κ (reversión)", 0.1, 10.0, 2.0, 0.1, key="h_kappa")
        h_theta = (st.slider("θ̄ vol largo plazo (%)", 1.0, 80.0, 20.0, 0.5, key="h_theta") / 100) ** 2
        h_xi    = st.slider("ξ (vol de vol)", 0.05, 1.5, 0.3, 0.05, key="h_xi")
        h_rho   = st.slider("ρ (correlación)", -0.99, 0.99, -0.7, 0.01, key="h_rho")
    with cm:
        st.markdown(f"<div style='font-size:0.72rem;color:{text_sub};margin-bottom:0.4rem'>Merton</div>", unsafe_allow_html=True)
        m_lam    = st.slider("λ (intensidad)", 0.0, 10.0, 1.0, 0.1, key="m_lam")
        m_mu_j   = st.slider("μⱼ (media salto)", -1.0, 1.0, -0.1, 0.01, key="m_mu_j")
        m_sigma_j = st.slider("σⱼ (vol salto)", 0.01, 0.8, 0.15, 0.01, key="m_sigma_j")

# ── Compute all models ──
with st.spinner("Calculando todos los modelos..."):
    results = compare_all_models(
        S, K, T, r, sigma, q,
        heston_params={"v0": sigma**2, "kappa": h_kappa, "theta": h_theta, "xi": h_xi, "rho": h_rho},
        merton_params={"lam": m_lam, "mu_j": m_mu_j, "sigma_j": m_sigma_j},
    )

bsm_ref_call = results["BSM"]["call"]
bsm_ref_put  = results["BSM"]["put"]

# ── Summary cards ──
st.markdown('<div class="section-header">Precios — Call Europeo</div>', unsafe_allow_html=True)
cols = st.columns(4)
for col, (model, color) in zip(cols, MODEL_COLORS.items()):
    price_c = results[model]["call"]
    diff    = price_c - bsm_ref_call
    pct     = diff / bsm_ref_call * 100 if bsm_ref_call != 0 else 0
    with col:
        st.markdown(f"""
        <div class="model-result-card" style="border-top: 3px solid {color}">
            <div class="model-name" style="color:{color}">{model}</div>
            <div class="model-price">${price_c:.4f}</div>
            <div class="model-delta" style="color:{'#34d399' if diff>=0 else '#f87171'}">
                {"+" if diff>=0 else ""}{diff:.4f} ({pct:+.2f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="section-header">Precios — Put Europeo</div>', unsafe_allow_html=True)
cols2 = st.columns(4)
for col, (model, color) in zip(cols2, MODEL_COLORS.items()):
    price_p = results[model]["put"]
    diff    = price_p - bsm_ref_put
    pct     = diff / bsm_ref_put * 100 if bsm_ref_put != 0 else 0
    with col:
        st.markdown(f"""
        <div class="model-result-card" style="border-top: 3px solid {color}">
            <div class="model-name" style="color:{color}">{model}</div>
            <div class="model-price">${price_p:.4f}</div>
            <div class="model-delta" style="color:{'#34d399' if diff>=0 else '#f87171'}">
                {"+" if diff>=0 else ""}{diff:.4f} ({pct:+.2f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

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

tab1, tab2, tab3 = st.tabs(["Comparativa por strike", "Sensibilidad a σ", "Heatmap de divergencia"])

# ─────────────────────────────
with tab1:
    strikes = np.linspace(max(S * 0.6, 1), S * 1.4, 60)
    opt_choice = st.radio("Tipo de opción", ["Call", "Put"], horizontal=True)
    oc = opt_choice.lower()

    with st.spinner("Calculando series por strike..."):
        bsm_s   = [BSMEngine(S, k, T, r, sigma, q).call_price()                  if oc=="call" else BSMEngine(S, k, T, r, sigma, q).put_price()   for k in strikes]
        crr_s   = [CRREngine(S, k, T, r, sigma, q, 200).call_price()              if oc=="call" else CRREngine(S, k, T, r, sigma, q, 200).put_price() for k in strikes]
        heston_s= []
        merton_s= []
        for k in strikes:
            h = HestonEngine(S, k, T, r, q, sigma**2, h_kappa, h_theta, h_xi, h_rho)
            m = MertonEngine(S, k, T, r, sigma, q, m_lam, m_mu_j, m_sigma_j)
            heston_s.append(h.call_price() if oc=="call" else h.put_price())
            merton_s.append(m.call_price() if oc=="call" else m.put_price())

    fig = go.Figure()
    for vals, model in zip([bsm_s, crr_s, heston_s, merton_s], MODEL_COLORS):
        fig.add_trace(go.Scatter(x=strikes, y=vals, name=model,
                                  line=dict(color=MODEL_COLORS[model], width=2 if model!="BSM" else 1.5,
                                             dash="dash" if model=="BSM" else "solid")))
    fig.add_vline(x=K, line_dash="dot", line_color=text_sub, annotation_text="K")
    fig.add_vline(x=S, line_dash="dot", line_color="#34d399", annotation_text="S")
    fig.update_layout(**plotly_layout(f"Precios {opt_choice} por Strike — todos los modelos"), height=400)
    fig.update_xaxes(title="Strike K")
    fig.update_yaxes(title=f"Precio {opt_choice}")
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────
with tab2:
    sigmas = np.linspace(0.05, 0.80, 50)
    with st.spinner("Sensibilidad a σ..."):
        bsm_sig   = [BSMEngine(S, K, T, r, s, q).call_price()               for s in sigmas]
        crr_sig   = [CRREngine(S, K, T, r, s, q, 200).call_price()           for s in sigmas]
        heston_sig= [HestonEngine(S, K, T, r, q, s**2, h_kappa, s**2, h_xi, h_rho).call_price() for s in sigmas]
        merton_sig= [MertonEngine(S, K, T, r, s, q, m_lam, m_mu_j, m_sigma_j).call_price()       for s in sigmas]

    fig2 = go.Figure()
    for vals, model in zip([bsm_sig, crr_sig, heston_sig, merton_sig], MODEL_COLORS):
        fig2.add_trace(go.Scatter(x=sigmas * 100, y=vals, name=model,
                                   line=dict(color=MODEL_COLORS[model], width=2)))
    fig2.add_vline(x=sigma * 100, line_dash="dot", line_color=text_sub, annotation_text="σ actual")
    fig2.update_layout(**plotly_layout("Sensibilidad al parámetro σ — Call ATM"), height=380)
    fig2.update_xaxes(title="σ (%)")
    fig2.update_yaxes(title="Precio Call")
    st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────
with tab3:
    if mode == "Avanzado":
        st.markdown(f"<div style='font-size:0.8rem;color:{text_sub};margin-bottom:0.8rem'>Diferencia porcentual de cada modelo vs BSM a través de una grilla de σ × T.</div>", unsafe_allow_html=True)
        model_heat = st.selectbox("Modelo a comparar vs BSM", ["CRR", "Heston", "Merton"])

        sigmas_h = np.linspace(0.05, 0.60, 15)
        mats_h   = np.linspace(0.1, 2.0, 15)
        Z = np.zeros((len(mats_h), len(sigmas_h)))

        with st.spinner(f"Calculando heatmap {model_heat} vs BSM..."):
            for i, t_h in enumerate(mats_h):
                for j, s_h in enumerate(sigmas_h):
                    bsm_p = BSMEngine(S, K, t_h, r, s_h, q).call_price()
                    if model_heat == "CRR":
                        other = CRREngine(S, K, t_h, r, s_h, q, 100).call_price()
                    elif model_heat == "Heston":
                        other = HestonEngine(S, K, t_h, r, q, s_h**2, h_kappa, s_h**2, h_xi, h_rho).call_price()
                    else:
                        other = MertonEngine(S, K, t_h, r, s_h, q, m_lam, m_mu_j, m_sigma_j).call_price()
                    Z[i, j] = (other - bsm_p) / bsm_p * 100 if bsm_p > 0.01 else 0.0

        fig_h = go.Figure(data=go.Heatmap(
            z=Z, x=sigmas_h * 100, y=mats_h,
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="% vs BSM"),
        ))
        fig_h.update_layout(**plotly_layout(f"Divergencia {model_heat} vs BSM (%) — grilla σ × T"), height=420)
        fig_h.update_xaxes(title="σ (%)")
        fig_h.update_yaxes(title="T (años)")
        st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.info("Activa el **Modo Avanzado** para ver el heatmap de divergencia σ × T.")
