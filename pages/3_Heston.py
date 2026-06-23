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

st.markdown('<div class="page-eyebrow">Vol Estocástica · Página 3</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Modelo de Heston (1993)</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Volatilidad estocástica con reversión a la media. Genera sonrisa de volatilidad de forma endógena, sin suponer σ constante.</div>', unsafe_allow_html=True)

#  Preset profiles 
PRESETS = {
    "Personalizado": None,
    "SPX típico (skew pronunciado)":   dict(kappa=3.0, theta_pct=20.0, xi=0.40, rho=-0.75),
    "Alta vol-de-vol (smile simétrico)": dict(kappa=1.5, theta_pct=25.0, xi=0.80, rho=-0.10),
    "Reversión rápida (term structure)": dict(kappa=8.0, theta_pct=18.0, xi=0.25, rho=-0.50),
    "Condición Feller límite":           dict(kappa=1.0, theta_pct=15.0, xi=0.55, rho=-0.60),
}
preset_choice = st.selectbox("Perfil de parámetros", list(PRESETS.keys()), index=0)

st.markdown(f"""
<div class="formula-box">
    <strong style="color:{text_main}">Dinámica estocástica de Heston</strong><br><br>
    dS = (r &minus; q)·S·dt + √v·S·dW<sub>1</sub><br>
    dv = κ(θ &minus; v)·dt + ξ·√v·dW<sub>2</sub><br>
    corr(dW<sub>1</sub>, dW<sub>2</sub>) = ρ &nbsp;&nbsp;&nbsp; [correlación entre retornos y varianza]<br><br>
    κ = velocidad de reversión &nbsp;·&nbsp; θ = varianza de largo plazo<br>
    ξ = vol de vol &nbsp;·&nbsp; v<sub>0</sub> = varianza inicial<br><br>
    <strong style="color:{text_main}">Condición de Feller:</strong> &nbsp; 2κθ &gt; ξ² &nbsp; (garantiza v(t) &gt; 0 a.s.)
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="background:{card};border:1px solid {border};border-radius:10px;padding:1.4rem 1.6rem;margin:0.8rem 0 1.2rem 0;line-height:1.85">
    <div style="font-size:0.7rem;font-family:JetBrains Mono,monospace;letter-spacing:0.14em;text-transform:uppercase;color:{accent};margin-bottom:0.9rem">
        Por qué Heston — el problema de la volatilidad constante
    </div>
    <p style="color:{text_main};font-size:0.88rem;margin:0 0 0.75rem 0">
        BSM asume que la volatilidad σ es constante en el tiempo. Pero en los mercados reales, la volatilidad
        <em>fluctúa</em>, exhibe clusters (periodos de alta vol seguidos de alta vol) y tiende a revertir a una media de largo plazo.
        Además, la volatilidad implícita extraída del mercado varía según el strike y la madurez —
        la llamada <strong style="color:{text_main}">sonrisa/skew de volatilidad</strong> — contradiciendo directamente el supuesto BSM.
        Heston (1993) fue la primera solución analítica tractable a este problema.
    </p>
    <p style="color:{text_main};font-size:0.88rem;margin:0 0 0.75rem 0">
        La innovación clave es modelar la varianza v(t) como un <strong style="color:{text_main}">proceso de Ornstein-Uhlenbeck
        con raíz cuadrada</strong> (proceso CIR): la varianza revierte hacia su media de largo plazo θ con velocidad κ,
        pero también tiene su propia fuente de aleatoriedad (ξ·√v·dW₂). Al correlacionar los movimientos del precio
        con los de la varianza mediante ρ, el modelo captura el <em>leverage effect</em> observado en acciones:
        cuando el precio cae, la volatilidad tiende a subir (ρ &lt; 0).
    </p>
    <div style="border-top:1px solid {border};margin:0.9rem 0"></div>
    <div style="font-size:0.7rem;font-family:JetBrains Mono,monospace;letter-spacing:0.14em;text-transform:uppercase;color:{accent};margin-bottom:0.6rem">
        Solución por función característica
    </div>
    <p style="color:{text_main};font-size:0.88rem;margin:0 0 0.75rem 0">
        Heston no tiene fórmula cerrada directa, pero sí una <strong style="color:{text_main}">solución semi-analítica</strong>
        mediante la función característica del log-precio. El precio del call se expresa como:
    </p>
    <div style="font-family:JetBrains Mono,monospace;font-size:0.82rem;color:{text_sub};padding:0.6rem 0.8rem;background:{surface};border-radius:6px;margin-bottom:0.75rem">
        C = S·e<sup>&minus;qT</sup>·P<sub>1</sub> &minus; K·e<sup>&minus;rT</sup>·P<sub>2</sub><br><br>
        P<sub>j</sub> = ½ + (1/π) · ∫<sub>0</sub><sup>∞</sup> Re[ e<sup>&minus;iφ ln K</sup> · φ<sub>j</sub>(φ) / (iφ) ] dφ
    </div>
    <p style="color:{text_sub};font-size:0.83rem;margin:0 0 0.4rem 0">
        donde φ<sub>j</sub>(φ) es la función característica del modelo (solución de la EDP de Heston en el espacio de Fourier).
        La integral se evalúa numéricamente — aquí usamos la formulación de Albrecher et al. (2007),
        que resuelve una discontinuidad en el corte de rama del logaritmo complejo presente en la formulación original de Heston.
    </p>
    <div style="border-top:1px solid {border};margin:0.9rem 0"></div>
    <div style="font-size:0.7rem;font-family:JetBrains Mono,monospace;letter-spacing:0.14em;text-transform:uppercase;color:{accent};margin-bottom:0.6rem">
        Interpretación de los parámetros
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem 1.5rem">
        <div style="font-size:0.83rem">
            <span style="color:{text_main};font-family:JetBrains Mono,monospace">v₀</span>
            <span style="color:{text_sub}"> — varianza inicial = σ₀². Si v₀ = θ, el proceso arranca en equilibrio.</span>
        </div>
        <div style="font-size:0.83rem">
            <span style="color:{text_main};font-family:JetBrains Mono,monospace">κ</span>
            <span style="color:{text_sub}"> — velocidad de reversión. κ alto → la vol vuelve rápido a θ. Medio: κ ≈ 2-5.</span>
        </div>
        <div style="font-size:0.83rem">
            <span style="color:{text_main};font-family:JetBrains Mono,monospace">θ</span>
            <span style="color:{text_sub}"> — varianza de largo plazo. Para σ̄ ≈ 20%, θ = 0.04. Ancla el nivel de vol.</span>
        </div>
        <div style="font-size:0.83rem">
            <span style="color:{text_main};font-family:JetBrains Mono,monospace">ξ</span>
            <span style="color:{text_sub}"> — vol de vol. Controla la curvatura de la sonrisa. ξ alto → smile más pronunciado.</span>
        </div>
        <div style="font-size:0.83rem">
            <span style="color:{text_main};font-family:JetBrains Mono,monospace">ρ</span>
            <span style="color:{text_sub}"> — correlación S-v. ρ &lt; 0 genera skew negativo (típico en acciones). ρ ≈ &minus;0.7 es común.</span>
        </div>
        <div style="font-size:0.83rem">
            <span style="color:{text_main};font-family:JetBrains Mono,monospace">Feller</span>
            <span style="color:{text_sub}"> — condición 2κθ &gt; ξ² garantiza v(t) &gt; 0 casi seguramente. Si se viola, la varianza puede tocar cero.</span>
        </div>
    </div>
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
    _p = PRESETS[preset_choice] or {}
    kappa     = st.slider("κ — velocidad de reversión", 0.1, 10.0, float(_p.get("kappa", 2.0)), 0.1)
    theta_pct = st.slider("θ — vol largo plazo (%)", 1.0, 80.0, float(_p.get("theta_pct", 20.0)), 0.5)
    theta = (theta_pct / 100) ** 2
    xi    = st.slider("ξ — vol de vol", 0.05, 1.5, float(_p.get("xi", 0.3)), 0.05)
    rho   = st.slider("ρ — correlación S-v", -0.99, 0.99, float(_p.get("rho", -0.7)), 0.01)

    feller = 2 * kappa * theta > xi**2
    feller_color = "#34d399" if feller else "#f87171"
    st.markdown(f"""
    <div style='background:{card};border:1px solid {feller_color}44;border-radius:6px;padding:0.6rem 1rem;margin-top:0.5rem;font-size:0.8rem'>
        <span style='color:{feller_color}'>{"" if feller else ""} Condición de Feller: 2κθ = {2*kappa*theta:.4f} {'>' if feller else '<'} ξ² = {xi**2:.4f}</span>
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

tab1, tab2, tab3, tab4 = st.tabs(["Sonrisa de volatilidad", "Superficie IV", "Sensibilidad a ρ y ξ", "Calibracion al mercado"])

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

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        show_multi_mat = st.checkbox("Superponer múltiples madureces", value=False)
    with col_s2:
        show_moneyness = st.checkbox("Eje en moneyness ln(K/S)", value=False)

    with st.spinner("Calculando sonrisa Heston..."):
        ivs_heston = h.vol_smile(strikes_smile)
    ivs_bsm_flat = np.full(len(strikes_smile), np.sqrt(v0))

    x_axis = np.log(strikes_smile / S) if show_moneyness else strikes_smile
    x_label = "Moneyness ln(K/S)" if show_moneyness else "Strike K"
    x_K = np.log(K/S) if show_moneyness else K

    fig = go.Figure()
    if show_multi_mat:
        mat_colors = {"0.25y": text_sub, "0.5y": "#f59e0b", "1y": accent, "2y": "#34d399"}
        for t_label, col_mat in mat_colors.items():
            t_val = {"0.25y": 0.25, "0.5y": 0.5, "1y": 1.0, "2y": 2.0}[t_label]
            h_t = HestonEngine(S, K, t_val, r, q, v0, kappa, theta, xi, rho)
            with st.spinner(f"Calculando T={t_label}..."):
                ivs_t = h_t.vol_smile(strikes_smile)
            fig.add_trace(go.Scatter(x=x_axis, y=ivs_t * 100, name=f"Heston T={t_label}",
                                      line=dict(color=col_mat, width=2 if t_label=="1y" else 1.5)))
    else:
        fig.add_trace(go.Scatter(x=x_axis, y=ivs_heston * 100, name="Heston IV", line=dict(color=accent, width=2.5)))
    fig.add_trace(go.Scatter(x=x_axis, y=ivs_bsm_flat * 100, name="BSM Flat IV",
                              line=dict(color="#f87171", width=1.5, dash="dash")))
    fig.add_vline(x=x_K, line_dash="dot", line_color=text_sub, annotation_text="K")
    fig.update_layout(**plotly_layout("Sonrisa de Volatilidad — Heston vs BSM"), height=400)
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title="Volatilidad Implícita (%)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    maturities = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    strikes_surf = np.linspace(S * 0.75, S * 1.25, 25)

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

with tab4:
    from engine import calibrate_heston
    st.markdown(f"<div style='font-size:0.85rem;color:{text_sub};margin-bottom:1rem'>Calibracion de los 5 parametros de Heston (kappa, theta, xi, rho, v0) a una sonrisa de mercado observada. Minimiza el RMSE sobre la IV implicita.</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:0.72rem;color:{text_sub};text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem'>Sonrisa de mercado (strike, IV%)</div>", unsafe_allow_html=True)

    n_cal = st.slider("Numero de strikes de mercado", 3, 10, 6, key="hcal_n")
    default_ks  = [S*m for m in [0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.30, 1.40, 1.50][:n_cal]]
    default_ivs = [24.0, 21.5, 20.5, 20.0, 20.5, 21.5, 23.5, 25.5, 27.0, 28.5][:n_cal]

    cal_strikes, cal_ivs = [], []
    cols_cal = st.columns(min(n_cal, 5))
    for i in range(n_cal):
        col = cols_cal[i % 5]
        with col:
            k_c  = col.number_input(f"K_{i+1}", value=float(round(default_ks[i], 1)), step=1.0, format="%.1f", key=f"hcal_k{i}")
            iv_c = col.number_input(f"IV_{i+1} (%)", value=float(default_ivs[i] if i < len(default_ivs) else 20.0),
                                     min_value=0.5, max_value=150.0, step=0.1, format="%.2f", key=f"hcal_iv{i}")
            cal_strikes.append(k_c)
            cal_ivs.append(iv_c / 100)

    st.markdown(f"<div style='font-size:0.72rem;color:{text_sub};text-transform:uppercase;letter-spacing:0.1em;margin:0.8rem 0 0.4rem'>Punto inicial para la optimizacion</div>", unsafe_allow_html=True)
    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
    kappa0 = cc1.number_input("kappa0", value=2.0, min_value=0.01, step=0.1, format="%.2f", key="hc_k0")
    theta0 = (cc2.number_input("theta0 (%)", value=20.0, min_value=0.1, step=0.5, format="%.1f", key="hc_t0") / 100)**2
    xi0    = cc3.number_input("xi0", value=0.3, min_value=0.01, step=0.05, format="%.3f", key="hc_xi0")
    rho0   = cc4.number_input("rho0", value=-0.5, min_value=-0.99, max_value=0.99, step=0.05, format="%.2f", key="hc_r0")
    v00    = (cc5.number_input("v0_0 (sigma0 %)", value=20.0, min_value=0.1, step=0.5, format="%.1f", key="hc_v0") / 100)**2

    if st.button("Calibrar Heston", type="primary"):
        with st.spinner("Calibrando (optimizacion L-BFGS-B)..."):
            cal_result = calibrate_heston(
                S, T, r, q,
                strikes=cal_strikes, market_ivs=cal_ivs,
                kappa0=kappa0, theta0=theta0, xi0=xi0, rho0=rho0, v00=v00,
            )

        status_color = "#34d399" if cal_result["success"] else "#f87171"
        st.markdown(f"""
        <div style='background:{card};border:1px solid {status_color}44;border-radius:8px;padding:0.8rem 1rem;margin-bottom:0.8rem;font-size:0.8rem'>
            <span style='color:{status_color}'>{'Calibracion exitosa' if cal_result["success"] else 'Convergencia parcial'}</span>
            &nbsp;&middot;&nbsp;
            <span style='color:{text_sub}'>RMSE: </span>
            <span style='font-family:"JetBrains Mono",monospace;color:{text_main}'>{cal_result["rmse"]*100:.4f}%</span>
            &nbsp;&middot;&nbsp;
            <span style='color:{text_sub}'>Mensaje: {cal_result["message"][:60]}</span>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("kappa", f"{cal_result['kappa']:.4f}")
        m2.metric("theta (vol LP)", f"{(cal_result['theta']**0.5)*100:.2f}%")
        m3.metric("xi", f"{cal_result['xi']:.4f}")
        m4.metric("rho", f"{cal_result['rho']:.4f}")
        m5.metric("sigma0", f"{(cal_result['v0']**0.5)*100:.2f}%")

        # Plot calibrated smile vs market
        import plotly.graph_objects as go
        h_cal = HestonEngine(S, K, T, r, q,
                             cal_result["v0"], cal_result["kappa"],
                             cal_result["theta"], cal_result["xi"], cal_result["rho"])
        strikes_fine = np.linspace(min(cal_strikes)*0.95, max(cal_strikes)*1.05, 80)
        with st.spinner("Calculando sonrisa calibrada..."):
            ivs_cal  = h_cal.vol_smile(strikes_fine) * 100
            ivs_init = h.vol_smile(strikes_fine) * 100

        def plotly_layout_cal(title=""):
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

        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(x=strikes_fine, y=ivs_cal, name="Heston calibrado",
                                      line=dict(color="#34d399", width=2.5)))
        fig_cal.add_trace(go.Scatter(x=strikes_fine, y=ivs_init, name="Heston inicial",
                                      line=dict(color=text_sub, width=1.5, dash="dash")))
        fig_cal.add_trace(go.Scatter(x=cal_strikes, y=[iv*100 for iv in cal_ivs],
                                      name="Mercado", mode="markers",
                                      marker=dict(color="#f59e0b", size=10, symbol="diamond")))
        fig_cal.update_layout(**plotly_layout_cal("Heston calibrado vs sonrisa de mercado"), height=400)
        fig_cal.update_xaxes(title="Strike K")
        fig_cal.update_yaxes(title="IV (%)")
        st.plotly_chart(fig_cal, use_container_width=True)

        # Residuals
        ivs_at_cal_strikes = [HestonEngine(S, k, T, r, q, cal_result["v0"], cal_result["kappa"],
                                            cal_result["theta"], cal_result["xi"], cal_result["rho"]).vol_smile([k])[0]*100
                               for k in cal_strikes]
        residuals = [m_iv*100 - model_iv for m_iv, model_iv in zip(cal_ivs, ivs_at_cal_strikes)]
        fig_res = go.Figure()
        fig_res.add_trace(go.Bar(x=cal_strikes, y=residuals,
                                  marker_color=["#34d399" if r >= 0 else "#f87171" for r in residuals],
                                  name="Residuo (mkt - modelo)"))
        fig_res.add_hline(y=0, line_color=text_sub, line_width=1)
        _rl = plotly_layout_cal("Residuos: Mercado - Heston calibrado (%)")
        _rl["height"] = 250
        _rl["showlegend"] = False
        fig_res.update_layout(**_rl)
        fig_res.update_xaxes(title="Strike K")
        fig_res.update_yaxes(title="Residuo (%)")
        st.plotly_chart(fig_res, use_container_width=True)
    else:
        st.info("Configura los puntos de mercado arriba y presiona **Calibrar Heston** para iniciar la optimizacion.")
