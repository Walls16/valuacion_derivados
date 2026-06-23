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

st.markdown('<div class="page-eyebrow">Jump-Diffusion · Página 4</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Modelo de Merton (1976)</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Proceso difusivo con saltos de Poisson log-normales. Captura eventos de cola, skew de volatilidad y discontinuidades de precio.</div>', unsafe_allow_html=True)

PRESETS_M = {
    "Personalizado": None,
    "Crash bias (acciones)":      dict(lam=1.0, mu_j=-0.15, sigma_j=0.20),
    "Saltos simétricos (FX)":     dict(lam=2.0, mu_j=0.0,   sigma_j=0.10),
    "Saltos frecuentes pequeños": dict(lam=5.0, mu_j=-0.04, sigma_j=0.06),
    "Saltos raros grandes":       dict(lam=0.3, mu_j=-0.30, sigma_j=0.25),
    "Sin saltos (→ BSM)":         dict(lam=0.0, mu_j=0.0,   sigma_j=0.01),
}
preset_m = st.selectbox("Perfil de saltos", list(PRESETS_M.keys()), index=0)

st.markdown(f"""
<div class="formula-box">
    <strong style="color:{text_main}">Dinámica del proceso jump-diffusion</strong><br><br>
    dS/S = (r &minus; λκ)·dt + σ·dW + (J&minus;1)·dN<br><br>
    N(t) ~ Poisson(λ) &nbsp;·&nbsp; λ = intensidad (saltos esperados por año)<br>
    ln(J) ~ N(μ, σ²) &nbsp;·&nbsp; κ = E[J&minus;1] = e<sup>μ + σ²/2</sup> &minus; 1<br><br>
    <strong style="color:{text_main}">Precio (serie de Poisson-BSM)</strong><br><br>
    C = &Sigma;<sub>n=0</sub><sup>&infin;</sup> [e<sup>&minus;λ'T</sup>(λ'T)<sup>n</sup>/n!] · BSM(S, K, T, r<sub>n</sub>, σ<sub>n</sub>)<br><br>
    r<sub>n</sub> = r &minus; λκ + n·(μ + σ²/2)/T &nbsp;·&nbsp; σ<sub>n</sub> = &radic;(σ² + n·σ²/T)
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="background:{card};border:1px solid {border};border-radius:10px;padding:1.4rem 1.6rem;margin:0.8rem 0 1.2rem 0;line-height:1.85">
    <div style="font-size:0.7rem;font-family:JetBrains Mono,monospace;letter-spacing:0.14em;text-transform:uppercase;color:{accent};margin-bottom:0.9rem">
        Por qué saltos — el problema de las colas pesadas
    </div>
    <p style="color:{text_main};font-size:0.88rem;margin:0 0 0.75rem 0">
        Un proceso de difusión puro como el GBM de BSM genera retornos normalmente distribuidos.
        Sin embargo, los mercados reales muestran <strong style="color:{text_main}">colas más pesadas que la normal</strong>
        (exceso de kurtosis) y movimientos bruscos discontinuos — quiebras, anuncios de ganancias, shocks macroeconómicos.
        Merton (1976) extendió BSM superponiendo un <strong style="color:{text_main}">proceso de Poisson compuesto</strong>
        al movimiento browniano: los saltos ocurren aleatoriamente con intensidad λ y tamaño aleatorio log-normal.
    </p>
    <p style="color:{text_main};font-size:0.88rem;margin:0 0 0.75rem 0">
        La estructura del modelo es elegante: el precio en t=T es el producto de un término difusivo (lognormal)
        y n términos de salto (lognormales independientes), donde n ~ Poisson(λT).
        Al condicionar en n, el precio sigue siendo lognormal — lo que permite que la solución sea
        un <strong style="color:{text_main}">promedio ponderado de precios BSM</strong>, uno por cada número posible de saltos.
        El peso de cada término es la probabilidad de Poisson de que ocurran exactamente n saltos.
    </p>
    <div style="border-top:1px solid {border};margin:0.9rem 0"></div>
    <div style="font-size:0.7rem;font-family:JetBrains Mono,monospace;letter-spacing:0.14em;text-transform:uppercase;color:{accent};margin-bottom:0.6rem">
        La derivación: ¿por qué es una serie?
    </div>
    <p style="color:{text_main};font-size:0.88rem;margin:0 0 0.75rem 0">
        El precio de la opción es E<sup>Q</sup>[e<sup>-rT</sup>·max(S<sub>T</sub>&minus;K, 0)].
        Se descompone condicionando en el número de saltos n que ocurrieron:
    </p>
    <div style="font-family:JetBrains Mono,monospace;font-size:0.82rem;color:{text_sub};padding:0.6rem 0.8rem;background:{surface};border-radius:6px;margin-bottom:0.75rem">
        C = &Sigma;<sub>n=0</sub><sup>&infin;</sup> P(N=n) · E[payoff | N=n]<br><br>
        P(N=n) = e<sup>&minus;λ'T</sup>(λ'T)<sup>n</sup>/n! &nbsp;·&nbsp; λ' = λ(1+κ)<br><br>
        E[payoff | N=n] = BSM con r<sub>n</sub> y σ<sub>n</sub> ajustados por los n saltos
    </div>
    <p style="color:{text_sub};font-size:0.83rem;margin:0 0 0.4rem 0">
        r<sub>n</sub> y σ<sub>n</sub> incorporan el efecto de los n saltos: aumentan la varianza total
        (σ<sub>n</sub> &gt; σ para n &gt; 0) y ajustan el drift para mantener la condición de martingala.
        La serie converge rápidamente — con 40-50 términos el error es &lt; 10<sup>&minus;12</sup>.
    </p>
    <div style="border-top:1px solid {border};margin:0.9rem 0"></div>
    <div style="font-size:0.7rem;font-family:JetBrains Mono,monospace;letter-spacing:0.14em;text-transform:uppercase;color:{accent};margin-bottom:0.6rem">
        Interpretación de los parámetros de salto
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem 1.5rem">
        <div style="font-size:0.83rem">
            <span style="color:{text_main};font-family:JetBrains Mono,monospace">λ</span>
            <span style="color:{text_sub}"> — intensidad. λ=1 → en promedio 1 salto por año. λ=0 → modelo reduce a BSM exactamente.</span>
        </div>
        <div style="font-size:0.83rem">
            <span style="color:{text_main};font-family:JetBrains Mono,monospace">μ</span>
            <span style="color:{text_sub}"> — tamaño medio del salto (en log). μ &lt; 0 → saltos en promedio negativos (crash bias).</span>
        </div>
        <div style="font-size:0.83rem">
            <span style="color:{text_main};font-family:JetBrains Mono,monospace">σ</span>
            <span style="color:{text_sub}"> — dispersión del salto. σ alto → saltos de tamaño muy variable, aumenta kurtosis.</span>
        </div>
        <div style="font-size:0.83rem">
            <span style="color:{text_main};font-family:JetBrains Mono,monospace">κ</span>
            <span style="color:{text_sub}"> — salto medio esperado = e<sup>μ+σ²/2</sup>&minus;1. Afecta el drift de compensación para no-arbitraje.</span>
        </div>
    </div>
    <div style="border-top:1px solid {border};margin:0.9rem 0"></div>
    <p style="color:{text_sub};font-size:0.83rem;margin:0">
        <strong style="color:{text_main}">Limitación principal:</strong> el mercado de Merton es incompleto —
        el riesgo de salto no se puede cubrir perfectamente con el subyacente.
        Esto implica que el precio de la opción no es único bajo no-arbitraje;
        se necesita un supuesto adicional sobre la prima de riesgo de salto.
        Merton asume que el riesgo de salto es idiosincrático (diversificable), lo que fija la medida Q.
    </p>
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
    _pm = PRESETS_M[preset_m] or {}
    lam     = st.slider("λ — intensidad de saltos (por año)", 0.0, 10.0, float(_pm.get("lam", 1.0)), 0.1)
    mu_j    = st.slider("μ — tamaño medio del salto (log)", -1.0, 1.0, float(_pm.get("mu_j", -0.1)), 0.01)
    sigma_j = st.slider("σ — volatilidad del salto", 0.01, 1.0, float(_pm.get("sigma_j", 0.15)), 0.01)
    n_terms = st.slider("Términos de la serie", 10, 100, 50, 5)

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

tab1, tab2, tab3 = st.tabs(["Sonrisa de volatilidad", "Efecto de λ y σ", "Trayectorias simuladas"])

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
        fig_s.add_vline(x=sigma_j * 100, line_dash="dot", line_color=text_sub, annotation_text="σ actual")
        fig_s.update_layout(**plotly_layout("Call vs Vol de salto σ"), height=300)
        fig_s.update_xaxes(title="σ (%)")
        fig_s.update_yaxes(title="Precio Call")
        st.plotly_chart(fig_s, use_container_width=True)

with tab3:
    st.markdown(f"<div style='font-size:0.8rem;color:{text_sub};margin-bottom:0.8rem'>Simulación de Monte Carlo de trayectorias bajo el proceso Merton jump-diffusion</div>", unsafe_allow_html=True)

    col_mc1, col_mc2, col_mc3 = st.columns(3)
    with col_mc1:
        n_paths = st.slider("Número de trayectorias", 5, 100, 30)
    with col_mc2:
        show_histogram = st.checkbox("Distribución terminal S(T)", value=True)
    with col_mc3:
        compare_gbm = st.checkbox("Superponer GBM puro (sin saltos)", value=True)
    n_steps_sim = 252
    dt_sim = T / n_steps_sim
    np.random.seed(42)

    fig_mc = go.Figure()
    times = np.linspace(0, T, n_steps_sim + 1)

    terminal_prices = []
    for i in range(n_paths):
        S_path = [S]
        for _ in range(n_steps_sim):
            s_curr = S_path[-1]
            diffusion = (r - q - lam * kappa_j - 0.5 * sigma**2) * dt_sim + sigma * np.sqrt(dt_sim) * np.random.randn()
            n_jumps = np.random.poisson(lam * dt_sim)
            jump = sum(np.random.normal(mu_j, sigma_j) for _ in range(n_jumps))
            S_path.append(s_curr * np.exp(diffusion + jump))
        terminal_prices.append(S_path[-1])
        color = f"rgba(79,142,247,{0.2 + 0.5*(i/n_paths)})" if dark else f"rgba(37,99,235,{0.2 + 0.5*(i/n_paths)})"
        fig_mc.add_trace(go.Scatter(x=times, y=S_path, mode="lines",
                                     line=dict(width=1, color=color), showlegend=False))

    if compare_gbm:
        np.random.seed(99)
        for i in range(min(n_paths, 15)):
            S_gbm = [S]
            for _ in range(n_steps_sim):
                S_gbm.append(S_gbm[-1] * np.exp((r - q - 0.5*sigma**2)*dt_sim + sigma*np.sqrt(dt_sim)*np.random.randn()))
            fig_mc.add_trace(go.Scatter(x=times, y=S_gbm, mode="lines",
                                         line=dict(width=0.8, color="#34d399", dash="dot"),
                                         showlegend=True if i==0 else False,
                                         name="GBM puro" if i==0 else None,
                                         opacity=0.5))

    fig_mc.add_hline(y=K, line_dash="dot", line_color="#f87171", annotation_text="K")
    fig_mc.add_hline(y=S, line_dash="dot", line_color="#34d399", annotation_text="S₀")
    fig_mc.update_layout(**plotly_layout(f"Trayectorias — Merton (λ={lam}, μ={mu_j}, σ={sigma_j})"), height=400)
    fig_mc.update_xaxes(title="Tiempo (años)")
    fig_mc.update_yaxes(title="Precio S(t)")
    st.plotly_chart(fig_mc, use_container_width=True)

    if show_histogram:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=terminal_prices, nbinsx=40,
            name="Merton S(T)", marker_color=accent, opacity=0.75,
        ))
        if compare_gbm:
            np.random.seed(77)
            gbm_terminal = [S * np.exp((r - q - 0.5*sigma**2)*T + sigma*np.sqrt(T)*np.random.randn()) for _ in range(500)]
            fig_hist.add_trace(go.Histogram(
                x=gbm_terminal, nbinsx=40,
                name="GBM S(T)", marker_color="#34d399", opacity=0.5,
            ))
        fig_hist.add_vline(x=K, line_dash="dot", line_color="#f87171", annotation_text="K")
        fig_hist.add_vline(x=S, line_dash="dot", line_color=text_sub, annotation_text="S₀")
        fig_hist.update_layout(**plotly_layout("Distribución de S(T) — Merton vs GBM"),
                                 barmode="overlay", height=300)
        fig_hist.update_xaxes(title="S(T)")
        fig_hist.update_yaxes(title="Frecuencia")
        kurt_mc = float(np.mean([(x/np.mean(terminal_prices) - 1)**4 for x in terminal_prices]) / np.var([x/np.mean(terminal_prices) for x in terminal_prices])**2)
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption(f"Kurtosis exceso Merton ≈ {kurt_mc - 3:.2f} (Normal = 0). Cola pesada → kurtosis positiva.")
